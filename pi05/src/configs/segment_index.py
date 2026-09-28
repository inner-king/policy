"""세그먼트(= lerobot 의 '에피소드')가 원본 에피소드 어디에 속하는지 되찾는다.

## 왜 필요한가

우리 데이터셋의 `meta/episodes.jsonl` 은 세그먼트를 0부터 다시 매긴 번호만
가지고 있어, **원본 에피소드가 무엇이었는지가 사라져 있다.** 원본 단위로
무언가 하려면(전체 재생, 데모 재생으로 시뮬레이터를 중간 상태까지 몰기 등)
`data/segments.parquet` 과 다시 맞춰야 한다.

## 어떻게 되찾나

`scripts/build_dataset.py` 가 한 일을 그대로 재현하면 순서가 맞는다:

1. `move to` 세그먼트를 **시드 0** 으로 25%만 남긴다 (`build_dataset.py:306-310`)
2. `task_index` 로 groupby 한 순서대로 쌓는다 (`build_dataset.py:338`)

그러면 i 번째 행이 lerobot 의 i 번째 세그먼트가 된다.

⚠️ **다만 66,892개 중 38개는 이웃과 순서가 뒤바뀌어 있다.** 빌드가 워커 12개로
병렬 처리돼서다. 그래서 지시문과 길이가 **둘 다** 맞는 행에만 `ok=True` 를
달고, 쓰려는 에피소드가 전부 통과할 때만 쓴다. 어긋난 행을 모르고 쓰면
엉뚱한 세그먼트의 액션을 재생하게 되는데 **에러가 나지 않는다.**
"""
import json
import pathlib

import numpy as np
import pandas as pd

REPO = "jiu121/behavior-cooking"
NAV_KEEP = 0.25          # build_dataset.py 의 --nav-keep 기본값
SEED = 0                 # build_dataset.py 의 default_rng(0)
CHUNK = 1000             # meta/info.json 의 chunks_size


def build_index(root: pathlib.Path) -> pd.DataFrame:
    """segments.parquet 각 행에 lerobot 위치(lb_idx/lb_start/lb_len)와 ok 를 붙인다."""
    seg = pd.read_parquet(root / "data/segments.parquet")
    rng = np.random.default_rng(SEED)
    nav = seg.skill == "move to"
    seg = seg[~(nav & (rng.random(len(seg)) >= NAV_KEEP))].reset_index(drop=True)
    order = pd.concat([tg for _, tg in seg.groupby("task_index")]).reset_index(drop=True)

    eps = [json.loads(l) for l in
           open(root / f"data/lerobot/{REPO}/meta/episodes.jsonl")]
    if len(order) != len(eps):
        raise RuntimeError(
            f"필터 재현 실패: segments {len(order):,} vs episodes.jsonl {len(eps):,}. "
            "build_dataset.py 의 --nav-keep/시드가 바뀌었는지 확인할 것")
    lengths = np.array([e["length"] for e in eps], dtype=np.int64)
    instr = [(e.get("tasks") or [""])[0] for e in eps]

    order["lb_idx"] = np.arange(len(order))                       # lerobot 에피소드 번호
    order["lb_start"] = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    order["lb_len"] = lengths
    order["ok"] = [order.instruction.iloc[i] == instr[i] and
                   int(order.n_frames.iloc[i]) == lengths[i] for i in range(len(order))]
    return order


def segment_parquet(root: pathlib.Path, lb_idx: int) -> pathlib.Path:
    """lerobot 세그먼트 번호 -> 그 세그먼트의 parquet 경로."""
    return (root / f"data/lerobot/{REPO}/data"
                   f"/chunk-{lb_idx // CHUNK:03d}/episode_{lb_idx:06d}.parquet")


def episode_view(order: pd.DataFrame, task_name: str, episode_index: int) -> pd.DataFrame:
    """원본 에피소드 하나의 세그먼트를 skill_idx 순으로."""
    g = order[(order.task_name == task_name) & (order.episode_index == episode_index)]
    return g.sort_values("skill_idx").reset_index(drop=True)


def pick_episode(order: pd.DataFrame, task_name: str, upto_skill: int) -> int:
    """0번부터 `upto_skill` 번까지가 **하나도 빠지지 않고** 남아 있는 에피소드를 고른다.

    이동 세그먼트를 75% 버렸기 때문에, 아무 에피소드나 고르면 재생 도중
    구멍이 생겨 로봇이 엉뚱한 곳에서 이어받게 된다. 연속성이 생명이다.
    """
    best = None
    for ep, g in order[order.task_name == task_name].groupby("episode_index"):
        g = g.sort_values("skill_idx")
        have = set(int(s) for s in g.skill_idx)
        if not set(range(upto_skill + 1)) <= have:
            continue
        head = g[g.skill_idx <= upto_skill]
        if not head.ok.all():
            continue
        frames = int(head.lb_len.sum())
        if best is None or frames < best[1]:
            best = (int(ep), frames)          # 짧을수록 재생이 빠르다
    if best is None:
        raise ValueError(f"{task_name} 에서 0~{upto_skill}번 스킬이 온전한 에피소드가 없다")
    return best[0]
