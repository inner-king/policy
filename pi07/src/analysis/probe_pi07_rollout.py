#!/usr/bin/env python3
"""롤아웃에서 LL 이 실제로 보낸 액션 vs 사람 데모의 액션 — 왜 팔이 움직이는데 아무 일도 안 일어나는가.

    python3 src/analysis/probe_pi07_rollout.py [eval-dir ...]

입력: runs/eval/<id>/actions-*/ep*.npy (서버가 기록한 23차원 원단위 액션) 와 data/pi07 의 데모 액션.
산출: runs/eval/<id>/analysis/rollout_actions.json
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

CAPSTONE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(CAPSTONE, "src", "pi07"))

GROUPS = [("베이스", 0, 3), ("토르소", 3, 7), ("왼팔", 7, 14), ("왼그리퍼", 14, 15),
          ("오른팔", 15, 22), ("오른그리퍼", 22, 23)]
EXEC = 25  # serve_pi07.py 가 청크당 실행한 스텝 수


def per_group(x: np.ndarray) -> dict:
    return {g: float(x[..., a:b].mean()) for g, a, b in GROUPS}


def stats(a: np.ndarray, exec_steps: int | None) -> dict:
    """a: [T, 23] 원단위 액션 시계열."""
    d = np.abs(np.diff(a, axis=0))
    sign_flip = (np.diff(np.sign(np.diff(a, axis=0)), axis=0) != 0).mean(0)  # 방향 전환 빈도 = 떨림
    out = {"T": int(a.shape[0]),
           "step_delta": per_group(d.mean(0)),
           "sign_flip_rate": per_group(sign_flip),
           "range": per_group(a.max(0) - a.min(0)),
           "grip_left_mean": float(a[:, 14].mean()), "grip_right_mean": float(a[:, 22].mean())}
    # 그리퍼 개폐 횟수: 자기 범위의 중간을 가로지른 횟수 (데모·롤아웃 각자의 척도로)
    for name, j in (("left", 14), ("right", 22)):
        v = a[:, j]
        mid = 0.5 * (v.max() + v.min())
        crossings = int((np.diff((v > mid).astype(np.int8)) != 0).sum())
        out[f"grip_{name}_crossings"] = crossings
        out[f"grip_{name}_frac_low"] = float((v <= mid).mean())
    if exec_steps:
        t = np.arange(1, a.shape[0])
        bnd = d[(t % exec_steps) == 0].mean(0)
        inr = d[(t % exec_steps) != 0].mean(0)
        out["chunk_boundary_delta"] = per_group(bnd)
        out["in_chunk_delta"] = per_group(inr)
    return out


def mean_dicts(ds: list[dict]) -> dict:
    out = {}
    for k in ds[0]:
        v = [d[k] for d in ds]
        out[k] = mean_dicts(v) if isinstance(v[0], dict) else float(np.mean(v))
    return out


def demo_stats(task: str, n: int = 10) -> dict:
    from dataset import B1KIndex
    idx = B1KIndex()
    eps = [e for e in idx.episodes if e["task"] == task][:n]
    rows = []
    for e in eps:
        a = np.asarray(idx.lowdim[e["offset"]:e["offset"] + e["length"], :23], np.float32)
        rows.append(stats(a, None))
    return mean_dicts(rows), [e["length"] for e in eps]


def main():
    dirs = sys.argv[1:] or [os.path.join(CAPSTONE, "runs", "eval", d) for d in
                            ("pi07ll-pi05base-30k-at30k-thawing_frozen_food-ll-hl",
                             "pi07ll-pi05base-30k-at30k-thawing_frozen_food-ll")]
    for d in dirs:
        # 디렉토리 이름 규약: <ll-run>-at<step>-<task>-<조건>
        task = os.path.basename(d).split("at30k-")[1].split("-ll")[0]
        rows = []
        for f in sorted(glob.glob(os.path.join(d, "actions-*", "ep*.npy"))):
            meta = json.load(open(f[:-4] + ".json"))
            a = np.load(f)
            if a.shape[0] < 1000:
                continue
            rows.append(stats(a, EXEC))
            rows[-1]["_file"] = os.path.relpath(f, d)
            rows[-1]["_complete"] = meta.get("complete")
        if not rows:
            print(f"{d}: 액션 로그 없음"); continue
        agg = mean_dicts([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
        dm, lens = demo_stats(task)
        out = {"eval": os.path.basename(d), "task": task, "n_rollouts": len(rows),
               "rollout": agg, "demo": dm, "demo_lengths": lens,
               "ratio_step_delta": {g: agg["step_delta"][g] / max(dm["step_delta"][g], 1e-9)
                                    for g, _, _ in GROUPS},
               "per_episode": rows}
        os.makedirs(os.path.join(d, "analysis"), exist_ok=True)
        p = os.path.join(d, "analysis", "rollout_actions.json")
        json.dump(out, open(p, "w"), indent=1, ensure_ascii=False)
        print(f"== {os.path.basename(d)}  n={len(rows)}  task={task}")
        print("  스텝당 변화량 배율 (롤아웃/데모):", {k: round(v, 2) for k, v in out["ratio_step_delta"].items()})
        print("  방향전환 비율 롤아웃:", {k: round(v, 3) for k, v in agg["sign_flip_rate"].items()})
        print("                 데모:", {k: round(v, 3) for k, v in dm["sign_flip_rate"].items()})
        print(f"  그리퍼 개폐 롤아웃 L/R {agg['grip_left_crossings']:.1f}/{agg['grip_right_crossings']:.1f}"
              f"  데모 {dm['grip_left_crossings']:.1f}/{dm['grip_right_crossings']:.1f}")
        print("  청크 경계 변화량:", {k: round(v, 4) for k, v in agg["chunk_boundary_delta"].items()})
        print("  청크 내부 변화량:", {k: round(v, 4) for k, v in agg["in_chunk_delta"].items()})
        print("  →", p)


if __name__ == "__main__":
    main()
