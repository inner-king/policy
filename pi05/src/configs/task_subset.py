"""지시문으로 데이터셋을 좁히는 장치 — 데이터를 다시 만들지 않고 인덱스만 거른다.

## 왜 필요한가

전체 데이터는 지시문이 466종, 2,658만 프레임이다. CU 예산으로 볼 수 있는
샘플은 수십만 개뿐이라, 전부를 대상으로 하면 **지시문 하나당 수백 번**밖에
못 본다. 실측한 이번 학습이 정확히 그랬다(320,000 / 466 = 687회).
그 상태의 정책은 오프라인 검사에서 '가만히 있기'와 거의 구분되지 않았다.

같은 CU 로 지시문을 20종으로 줄이면 종류당 16,000회, 5종으로 줄이면
64,000회가 된다. **데이터가 남아도는 상황에서는 범위를 좁히는 것이
곧 학습량을 늘리는 것**이다.

## 어떻게 거르나

새 데이터셋을 만들지 않는다. `meta/episodes.jsonl` 로 에피소드별 지시문과
길이를 읽어 프레임 인덱스 구간을 계산하고, 조건에 맞는 구간만 남긴
`torch.utils.data.Subset` 으로 감싼다. 디스크를 건드리지 않으므로 비용이 없다.

openpi 의 `create_torch_dataset` 하나만 감싸면 **학습과 정규화 통계가 둘 다**
같은 부분집합을 보게 된다(`compute_norm_stats.py:32` 도 같은 함수를 쓴다).

## 정규화 통계는 반드시 다시 계산해야 한다

부분집합은 액션 분포가 다르다. 전체로 낸 q01/q99 를 그대로 쓰면 정규화가
어긋나는데 **에러가 나지 않는다.** 그래서 부분집합마다 `asset_id` 를 따로
두고, 그 이름으로 필터를 찾는다 — 통계 파일 경로와 필터가 한 몸이 되어
서로 어긋날 수 없다.
"""
import json
import os
import pathlib
import zlib

import numpy as np
import torch.utils.data


class _Subset(torch.utils.data.Dataset):
    """인덱스를 numpy 로 들고 있되 **파이썬 int 로 변환해서** 넘기는 부분집합.

    `torch.utils.data.Subset` 을 그대로 쓰면 numpy 배열에서 꺼낸 `np.int64` 가
    그대로 내려가 HuggingFace datasets 가 거부한다:

        TypeError: Wrong key type: '19187153' of type '<class 'numpy.int64'>'.
        Expected one of int, slice, range, str or Iterable.

    인덱스를 파이썬 리스트로 두면 이 문제는 없지만, 727만 개짜리 리스트는
    int 객체만 200MB 를 넘고 openpi 의 DataLoader 는 **spawn** 워커라 그걸
    워커마다 피클해 보낸다. numpy 배열이면 버퍼 하나(58MB)로 끝난다.
    그래서 저장은 numpy, 접근은 int 로 한다.
    """

    def __init__(self, dataset, indices: np.ndarray):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i):
        return self.dataset[int(self.indices[i])]

# asset_id -> (모드, 값, 상한). 모드는 "exact"(지시문 전체 일치) 또는 "contains".
#
# 재학습에는 **exact** 를 쓴다. 지시문 목록을 문자열 그대로 박아 두면 나중에
# 데이터가 바뀌어도 무엇으로 학습했는지가 코드에 남는다. "contains" 는 탐색용이다.
FILTERS: dict[str, tuple[str, tuple[str, ...], "int | None"]] = {}
_applied = False


def register(asset_id: str, values, mode: str = "exact", cap: int | None = None) -> None:
    """`cap` 은 지시문 하나가 가져갈 수 있는 최대 프레임 수다.

    ## 왜 상한이 필요한가

    샘플은 프레임 수에 비례해 뽑힌다. 상위 20종 안에서도 편차가 26.8배라
    (`open the door of the fridge` 25.70% vs `chop the half bell pepper` 0.96%)
    가장 못 배운 지시문이 학습의 1% 도 못 받는다.

    **종류를 줄이는 것으로는 이 문제가 안 풀린다.** 상위 10종이 이미 프레임의
    81.5% 라, 하위 10종을 다 버려도 남는 것의 노출은 1.23배밖에 안 오른다.
    치우침 자체를 없애야 한다. 근거는 docs/HANDOFF.md 15.9.

    상한 100,000 이면 194만 프레임이 남고 최대/최소가 1.44배가 된다. 완전 균등
    (최소값 69,606 으로 전부 잘라내기)은 데이터의 81% 를 버리므로 쓰지 않는다.

    ⚠️ **가중 인덱스로 복제하면 안 된다.** 인덱스 배열은 spawn 워커마다 피클되어
    전달되므로(12.11) 복제는 그 비용을 키운다. 잘라내는 쪽이 싸다.
    """
    assert mode in ("exact", "contains"), mode
    FILTERS[asset_id] = (mode, tuple(values), cap)


def _frame_ranges(repo_id: str):
    """에피소드별 (지시문, 시작프레임, 끝프레임)."""
    root = pathlib.Path(os.environ["HF_LEROBOT_HOME"]) / repo_id
    start = 0
    for line in open(root / "meta/episodes.jsonl"):
        d = json.loads(line)
        task = (d.get("tasks") or [""])[0]
        yield task, start, start + d["length"]
        start += d["length"]


def _subset(dataset, repo_id: str, mode: str, values: tuple[str, ...], cap: int | None = None):
    want = set(values) if mode == "exact" else None
    per_task: dict[str, list[tuple[int, int]]] = {}
    for task, a, b in _frame_ranges(repo_id):
        t = task or ""
        hit = (t in want) if want is not None else any(v.lower() in t.lower() for v in values)
        if hit:
            per_task.setdefault(t, []).append((a, b))
    if not per_task:
        raise ValueError(f"조건에 맞는 에피소드가 없다 ({mode}) — 지시문 문자열을 확인할 것")
    if want is not None and len(per_task) != len(want):
        # 오타나 데이터 변경으로 조용히 일부가 빠지는 것을 막는다.
        raise ValueError(
            f"지시문 {len(want)}개를 지정했는데 {len(per_task)}개만 찾았다. "
            f"누락: {sorted(want - set(per_task))[:3]}"
        )

    keep, dropped = [], 0
    for t in sorted(per_task):                      # 정렬로 실행 간 재현성을 준다
        rs = per_task[t]
        total = sum(b - a for a, b in rs)
        if cap is None or total <= cap:
            keep += rs
            continue
        # 에피소드 순서를 지시문마다 고정 시드로 섞은 뒤 상한까지만 담는다.
        # 파일 순서 그대로 앞에서 자르면 원본 태스크가 한쪽으로 쏠릴 수 있다.
        # ⚠️ 파이썬 내장 hash() 는 문자열에 대해 프로세스마다 값이 달라진다
        # (PYTHONHASHSEED 무작위화). 그걸 쓰면 통계 계산과 학습이, 심지어 spawn
        # 워커끼리도 **서로 다른 부분집합**을 보게 되는데 에러가 나지 않는다.
        # zlib.crc32 는 어디서 돌려도 같은 값이다.
        order = np.random.default_rng(zlib.crc32(t.encode())).permutation(len(rs))
        acc = 0
        for i in order:
            a, b = rs[i]
            keep.append((a, b))
            acc += b - a
            if acc >= cap:                          # 에피소드 하나 길이만큼 초과할 수 있다
                break
        dropped += total - acc

    idx = np.concatenate([np.arange(a, b, dtype=np.int64) for a, b in sorted(keep)])
    msg = (f"[task_subset] 지시문 {len(per_task)}종 / {len(idx):,} 프레임 "
           f"(전체 {len(dataset):,} 의 {len(idx)/len(dataset)*100:.1f}%, "
           f"인덱스 {idx.nbytes/1e6:.0f}MB)")
    if cap is not None:
        msg += f" / 상한 {cap:,} 적용해 {dropped:,} 프레임 제외"
    print(msg, flush=True)
    return _Subset(dataset, idx)


def apply() -> None:
    """openpi 의 create_torch_dataset 을 한 번만 감싼다."""
    global _applied
    if _applied:
        return
    from openpi.training import data_loader as _dl

    orig = _dl.create_torch_dataset

    def wrapped(data_config, action_horizon, model_config):
        ds = orig(data_config, action_horizon, model_config)
        f = FILTERS.get(data_config.asset_id)
        return _subset(ds, data_config.repo_id, *f) if f else ds

    _dl.create_torch_dataset = wrapped
    _applied = True
