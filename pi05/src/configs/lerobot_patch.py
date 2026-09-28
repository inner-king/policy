"""6.7만 에피소드 데이터셋에서 lerobot 로더가 느린 세 곳을 우회한다.

lerobot 은 수천 에피소드를 가정하고 만들어져 있어서, 6.7만 개에서는 여러 군데가
**에피소드 수 또는 parquet 파일 수에 비례하는 비용**을 반복해서 낸다. 이미
쓰기 쪽에서 같은 병(`save_episode()` 의 O(N²))을 만나 우회했고, 읽기 쪽에도
세 자리가 있었다. openpi 와 lerobot 은 손대지 않고, 안정적인 이음매만 감싼다.

    import lerobot_patch; lerobot_patch.apply()

## 실측 요약 (66,892 에피소드 / 26,579,202 프레임)

| | 원래 | 우회 후 |
|---|---:|---:|
| 데이터셋 생성 | 321초 / 최대 RSS 25GB | **약 90~190초 / 11GB** |
| 샘플 하나 꺼내기 | 256.5 ms | **6.2 ms** (41배) |
| 단일 프로세스 처리량 | 3.9 샘플/초 | **161 샘플/초** |

배치 256 기준으로 워커 8개면 한 스텝 0.20초라 데이터 로딩이 더 이상 병목이
아니게 된다(원래대로면 워커 8개로도 한 스텝 8.3초, 30k 스텝에 69시간).

## (1) 기동: 타임스탬프 검사가 2,658만 행을 파이썬 텐서로 만든다

`LeRobotDataset.__init__` 은 마지막에 프레임 간격이 1/fps 인지 검사한다
(`lerobot_dataset.py:507-511`):

    timestamps      = torch.stack(self.hf_dataset["timestamp"]).numpy()
    episode_indices = torch.stack(self.hf_dataset["episode_index"]).numpy()

`hf_dataset` 에는 `set_transform(hf_transform_to_torch)` 가 걸려 있어 컬럼 하나를
꺼내는 순간 2,658만 행이 전부 0차원 텐서로 만들어진다. 321초 중 약 224초,
그리고 RSS 25GB 의 대부분이 여기다.

→ `load_hf_dataset` 이 돌려주는 객체에 얇은 껍데기(`_FastColumns`)를 씌워
  그 두 컬럼만 Arrow 에서 통째로 꺼낸다. 그러면 위 두 줄의 `torch.stack` 에
  리스트가 아니라 **이미 쌓인 텐서**가 들어가므로, `torch.stack` 도 텐서면
  그대로 돌려주도록 바꾼다.

  ⚠️ `m.torch` 는 전역 `torch` 모듈 **그 자체**다. 따라서 아래의
  `m.torch.stack = _stack` 은 모듈 하나가 아니라 **프로세스 전역의
  `torch.stack` 을 바꾼다.** 무해한 이유는 가로채는 조건이 좁기 때문이다 —
  인자가 Tensor 일 때만 그대로 돌려주고, 시퀀스를 넘기는 정상 호출
  (`default_collate` 포함)은 전부 원본 `torch.stack` 으로 빠진다.
  v1 학습 10,000 스텝이 이 상태로 완주한 것이 실측 근거다.

**검사를 없앤 것이 아니다.** 비싼 것은 검사가 아니라 값을 꺼내는 방식이었으므로
`check_timestamps_sync` 는 그대로 돈다 — 일부러 틀린 타임스탬프를 넣으면
여전히 `ValueError` 로 잡히는 것을 확인했다.

## (2) 샘플마다: 파일 6.7만 개의 체크섬을 deepcopy 한다

`__getitem__` 은 액션 청크를 얻으려 `hf_dataset.select()` 를 부르고, 그 안에서
`DatasetInfo.copy()` 가 `download_checksums` 를 deepcopy 한다. 이 dict 는
`load_dataset("parquet", ...)` 이 읽은 **파일 하나당 한 항목**이라 6.7만 개다.
샘플 하나에 deepcopy 가 80만 번 일어난다(40샘플 프로파일에서 3,200만 번,
전체 시간의 91%). 256.5 ms -> 71.4 ms.

→ 출처 기록용 메타데이터일 뿐이므로 로드 직후 지운다.
   (파일 1,000개 vs 1개로 따로 재보니 3.15ms vs 0.55ms 였고, 체크섬을 지우면
    둘 다 0.54ms 로 같아진다. 즉 parquet 을 합치는 것과 효과가 같다.)

## (3) 샘플마다: 액션 청크 조회가 비싼 select 경로를 탄다

남은 71.4 ms 의 대부분은 `_query_hf_dataset` 이다:

    torch.stack(self.hf_dataset.select(q_idx)[key])

`q_idx` 가 연속 범위가 아니라 **인덱스 리스트**라 `datasets` 가
`_select_with_indices_mapping` 으로 간다. 여기서 인덱스 매핑 테이블을 새로 만들고
`IndexedTableMixin` 이 7만 개 Arrow 청크를 다시 훑는다. 실측으로
`select(range(...))` 는 3.3ms 인데 전체 `__getitem__` 은 92ms 였다.

→ Arrow 테이블에서 `fast_gather` 로 직접 뽑는다. 71.4 ms -> 6.2 ms.
   원래 경로와 `actions` / `state` / `timestamp` / `episode_index` /
   `actions_is_pad` 가 dtype·shape·값까지 전부 일치하는 것을 확인했다.

## (4) 워커는 fork 가 아니라 spawn 이다 — 이걸 놓치면 위 셋이 전부 무효다

openpi 의 `TorchDataLoader` 는 `multiprocessing.get_context("spawn")` 을 쓴다
(`data_loader.py:429-430`). spawn 은 부모의 메모리를 물려받지 않고 **모듈을 새로
import 한다.** 그래서 부모에서 monkey-patch 를 걸어도 워커에는 적용되지 않고,
실제 학습/통계 계산은 패치 이전 속도(256ms/샘플)로 돈다. 배치 256 하나에 26초가
찍혀서 발견했다(기대값 0.85초).

게다가 spawn 은 데이터셋 객체를 **pickle 해서** 워커로 보내는데, (1)의 껍데기가
붙어 있으면 pickle 이 깨진다 — 예전에 `DataLoader worker exited unexpectedly` 로
죽던 것이 이것이었다.

→ 두 가지를 한다.
  - `LeRobotDataset.__init__` 이 끝나면 껍데기를 벗겨 원래 객체로 되돌린다
    (껍데기는 `__init__` 동안에만 필요하다).
  - openpi 의 `_worker_init_fn` 을 우리 것으로 바꿔 워커 시작 시 `apply()` 를
    부르고, 이어서 원래 함수를 호출한다. pickle 이 이름으로 직렬화하도록
    모듈 최상위 함수로 두고, 워커가 이 모듈을 import 할 수 있게
    `PYTHONPATH` 에 `configs/` 를 넣는다.

## 근본 원인과 정공법

셋 다 뿌리는 하나다 — **에피소드마다 parquet 파일을 하나씩 쓰는 LeRobot v2.1
레이아웃.** 원본 BEHAVIOR 데이터셋이 v3.0 에서 여러 에피소드를 한 파일에 모으는
이유가 이것이다. parquet 을 합쳐 다시 쓰는 것이 정공법이지만 포맷을 벗어나므로,
여기서는 읽는 쪽만 감쌌다. (2)의 실측이 보여 주듯 효과는 같다.
"""

import os
import pathlib

import numpy as np
import torch

_applied = False
_real_stack = torch.stack
_CFG_DIR = str(pathlib.Path(__file__).resolve().parent)
_ORIG_WORKER_INIT = None


class _FastColumns:
    """`ds["timestamp"]` 만 Arrow 에서 바로 꺼내고 나머지는 원본에 위임한다."""

    _FAST = ("timestamp", "episode_index")

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def __getitem__(self, key):
        if isinstance(key, str) and key in self._FAST:
            col = self._inner.data.column(key)
            return torch.from_numpy(np.asarray(col.to_numpy(zero_copy_only=False)))
        return self._inner[key]

    def __getattr__(self, name):
        # _inner 는 object.__setattr__ 로 넣었으므로 여기 오지 않는다 (무한 재귀 없음).
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name, value):
        setattr(self._inner, name, value)

    def __len__(self):
        return len(self._inner)

    def __iter__(self):
        return iter(self._inner)


def _stack(x, *a, **kw):
    if isinstance(x, torch.Tensor):     # _FastColumns 가 이미 쌓인 텐서를 준다
        return x
    return _real_stack(x, *a, **kw)


def _inner_of(hf_dataset):
    return getattr(hf_dataset, "_inner", hf_dataset)


def _query_hf_dataset(self, query_indices):
    """`select()` 대신 Arrow 에서 직접 행을 모은다 (원래 함수와 결과가 동일)."""
    tbl = _inner_of(self.hf_dataset).data
    out = {}
    for key, q in query_indices.items():
        sub = tbl.fast_gather(np.asarray(q, dtype=np.int64))
        a = sub.column(key).to_numpy(zero_copy_only=False)
        if a.dtype == object:                       # FixedSizeList -> 배열의 배열
            a = np.stack(a)
        out[key] = torch.from_numpy(np.ascontiguousarray(a))
    return out


def _worker_init(worker_id):
    """DataLoader 워커(spawn) 안에서 패치를 적용한다. (4) 참조.

    모듈 최상위 함수여야 pickle 이 이름 참조로 직렬화한다 — 클로저면 실패한다.
    """
    apply()
    if _ORIG_WORKER_INIT is not None:
        _ORIG_WORKER_INIT(worker_id)


def apply() -> None:
    """import 순서와 무관하게 한 번만 적용된다. 워커 안에서도 다시 불린다."""
    global _applied, _ORIG_WORKER_INIT
    if _applied:
        return
    from lerobot.common.datasets import lerobot_dataset as m

    orig_load = m.LeRobotDataset.load_hf_dataset
    orig_init = m.LeRobotDataset.__init__

    def load_hf_dataset(self):
        d = orig_load(self)
        try:
            d.info.download_checksums = None        # (2) 파일 6.7만 개의 체크섬
        except Exception:
            pass
        return _FastColumns(d)                      # (1) 컬럼 통째로 꺼내기

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        # (1)의 껍데기는 __init__ 동안에만 필요하다. 남겨 두면 spawn 워커로 보낼 때
        # pickle 이 깨진다(예전에 "DataLoader worker exited unexpectedly" 가 났던 원인).
        if isinstance(getattr(self, "hf_dataset", None), _FastColumns):
            self.hf_dataset = self.hf_dataset._inner

    m.LeRobotDataset.load_hf_dataset = load_hf_dataset
    m.LeRobotDataset.__init__ = __init__
    m.LeRobotDataset._query_hf_dataset = _query_hf_dataset   # (3) 액션 청크 조회
    # ⚠️ `m.torch` 는 전역 torch 모듈 그 자체라 이 줄은 프로세스 전역에 걸린다.
    # 좁게 가로채므로 무해하다 — 위 (1) 설명 참조.
    m.torch.stack = _stack

    # (4) spawn 워커에도 적용되게 한다
    pp = os.environ.get("PYTHONPATH", "")
    if _CFG_DIR not in pp.split(os.pathsep):
        os.environ["PYTHONPATH"] = (_CFG_DIR + os.pathsep + pp).rstrip(os.pathsep)
    try:
        import openpi.training.data_loader as dl
        cur = getattr(dl, "_worker_init_fn", None)
        if cur is not None and cur is not _worker_init:
            _ORIG_WORKER_INIT = cur
        dl._worker_init_fn = _worker_init
    except Exception:
        pass

    _applied = True
