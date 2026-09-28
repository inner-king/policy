"""데모의 관절 액션을 시뮬레이터 쪽에서 읽을 수 있는 파일로 내보낸다.

    third_party/openpi/.venv/bin/python scripts/export_demo_actions.py \
        --task slicing_vegetables [--episode 8652]

## 왜 필요한가

정책은 **이동을 못 한다.** 2차 학습은 지시문 상위 20종만 썼는데 `move to ...`
는 거기 없어서, 한 번도 본 적이 없다. 그런데 학습된 스킬 10개는 전부 태스크
시작점이 아니라 중간에 있다 (예: `open the door of the fridge` 는 618프레임짜리
이동 뒤에 온다).

그래서 시뮬레이터를 **데모의 녹화된 액션으로 그 지점까지 몰고 간 뒤** 정책에
넘긴다. 사람이 고속도로 진입까지 운전해 주고 거기서 자율주행으로 바꾸는 것과
같다.

## 왜 파일로 주고받나

시뮬레이터는 conda `behavior` 환경에서 돌고, 데이터셋을 읽는 코드는
`third_party/openpi/.venv` 에 있다. 한 프로세스에 안 들어가므로 npz 로 넘긴다.

## 값은 그대로 쓸 수 있다

액션 23차원(base 3 / trunk 4 / 왼팔 7 / 왼그리퍼 1 / 오른팔 7 / 오른그리퍼 1)의
구성과 순서가 시뮬레이터 컨트롤러와 같고, `use_delta_commands: false` 라
**절대 관절값**을 그대로 받는다. 변환이 필요 없다.
"""
import argparse, json, pathlib, sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "configs"))
import segment_index as si
from cooking_config import TOP20

TOP = set(TOP20)

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--task", default="slicing_vegetables")
ap.add_argument("--episode", type=int, default=None,
                help="원본 에피소드 번호. 생략하면 앞부분이 온전한 것 중 가장 짧은 것")
ap.add_argument("--upto-skill", type=int, default=1,
                help="이 스킬까지 연속으로 남아 있는 에피소드를 고를 때의 기준")
ap.add_argument("--out", default="outputs/demo")
ap.add_argument("--raw-dir", default="data/raw_demo",
                help="원본 데모 parquet 을 받아 둔 곳. 있으면 이동 구간까지 메운다")
a = ap.parse_args()


def raw_frame_source(root, task, episode):
    """원본 parquet 이 있으면 (DataFrame, 세그먼트표) 를 돌려준다. 없으면 None.

    우리 데이터셋은 `move to` 세그먼트의 75%를 버렸기 때문에, 그것만으로는
    재생이 중간중간 끊긴다 — 끊기면 로봇이 이동을 못 해 그 뒤가 전부 무의미하다.
    원본 액션 parquet(에피소드 하나당 78MB, 영상은 불필요)을 받아 두면 빠진
    구간을 메워 **처음부터 끝까지 이어지는** 재생을 만들 수 있다.

    원본과 우리 데이터의 액션·state 가 **비트 단위로 같음을 확인**했으므로
    (2026-08-29, 최대오차 0.00e+00) 섞어 써도 어긋나지 않는다.
    """
    seg = pd.read_parquet(root / "data/segments.parquet")
    g = seg[(seg.task_name == task) & (seg.episode_index == episode)].sort_values("skill_idx")
    if g.empty:
        return None
    r = g.iloc[0]
    f = (root / a.raw_dir / "data" /
         f"chunk-{int(r['data/chunk_index']):03d}" /
         f"file-{int(r['data/file_index']):03d}.parquet")
    if not f.exists():
        return None
    raw = pd.read_parquet(f).set_index("index")
    lo, hi = int(r.dataset_from_index), int(r.dataset_to_index)
    if raw.index.min() > lo or raw.index.max() < hi - 1:
        print(f"⚠️  원본 파일이 에피소드 구간을 다 담지 못한다 — 무시하고 진행")
        return None
    return raw, g


def main():
    order = si.build_index(ROOT)
    ep = a.episode if a.episode is not None else si.pick_episode(order, a.task, a.upto_skill)
    v = si.episode_view(order, a.task, ep)
    if not v.ok.all():
        # 순서가 뒤바뀐 행이 섞이면 엉뚱한 세그먼트의 액션을 재생하게 된다.
        bad = v[~v.ok].skill_idx.tolist()
        print(f"⚠️  정렬 검증에 실패한 스킬이 있다: {bad} — 그 지점 이후는 신뢰할 수 없다")

    src = raw_frame_source(ROOT, a.task, ep)
    if src is not None:
        raw, v = src                                 # 원본 기준 = 스킬이 하나도 안 빠진다
        print(f"원본 parquet 사용 — 스킬 {len(v)}개 (이동 구간 포함, 끊김 없음)\n")
    else:
        raw = None
        print(f"원본 parquet 없음 — 우리 데이터셋만 사용 (이동 구간이 빠져 끊긴다)\n")

    acts, states, bounds, t = [], [], [], 0
    prev = None
    for _, r in v.iterrows():
        if raw is not None:
            sl = raw.loc[int(r.abs_start):int(r.abs_end) - 1]
            A = np.stack(sl["action"].to_numpy())
            S = np.stack(sl["observation.state"].to_numpy())
        else:
            df = pd.read_parquet(si.segment_parquet(ROOT, int(r.lb_idx)))
            A = np.stack(df["actions"].to_numpy())      # (L, 23)
            S = np.stack(df["state"].to_numpy())        # (L, 61)
            assert len(A) == int(r.lb_len), f"길이 불일치 {len(A)} vs {r.lb_len}"
        gap = prev is not None and int(r.skill_idx) != prev + 1
        prev = int(r.skill_idx)
        bounds.append({"skill_idx": int(r.skill_idx), "instruction": r.instruction,
                       "trained": bool(r.instruction in TOP),
                       "t0": t, "t1": t + len(A), "n": len(A),
                       "gap_before": bool(gap)})
        acts.append(A); states.append(S); t += len(A)

    acts = np.concatenate(acts).astype(np.float32)
    states = np.concatenate(states).astype(np.float32)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.task}_ep{ep}.npz"
    np.savez_compressed(path, actions=acts, states=states,
                        boundaries=json.dumps(bounds, ensure_ascii=False))

    print(f"원본 에피소드 {ep} ({a.task})")
    print(f"  세그먼트 {len(bounds)}개 / {t:,}프레임 / 학습된 스킬 "
          f"{sum(b['trained'] for b in bounds)}개")
    print(f"  액션 {acts.shape} / state {states.shape}")
    print(f"  -> {path}  ({path.stat().st_size/1e6:.1f} MB)\n")
    print(f"{'스킬':>4} {'':2} {'지시문':<52} {'프레임구간':>16}")
    for b in bounds:
        mark = "✅" if b["trained"] else "⬜"
        gap = "  ← 이동 세그먼트 빠짐" if b["gap_before"] else ""
        print(f"{b['skill_idx']:4d} {mark} {b['instruction'][:52]:<52} "
              f"{b['t0']:7,}~{b['t1']:<7,}{gap}")


if __name__ == "__main__":
    main()
