"""문장만 바꿔 돌린 rollout 들의 관절 명령을 서로 비교한다 (HANDOFF 14.1 과 같은 잣대).

    python scripts/compare_rollout_actions.py outputs/rollout/*/actions_*.npz

## 왜 대조군이 반드시 필요한가

이 정책은 확률적이라 **같은 문장을 두 번 줘도 결과가 다르다.** 그 기준선을
모르면 "문장을 바꿔서 달라졌다" 를 말할 수 없다. 2026-08-29 에 대조군 없이
비교했다가 하마터면 반대 결론을 낼 뻔했다(HANDOFF 14.1).

그래서 **같은 문장 두 번**을 1.00 배로 놓고, 다른 문장들이 그보다 얼마나 더
벌어지는지를 본다.

## 무엇을 비교하나

팔·토르소 18 차원만 본다. 베이스 3 차원은 `--zero-base` 로 눌러 놓아 정보가
없고, 그리퍼 2 차원은 값의 폭이 달라 평균을 지배한다.
"""
import itertools
import pathlib
import sys

import numpy as np

# 팔·토르소 = 액션 23 차원 중 베이스(0~2) 와 그리퍼(14, 22) 를 뺀 18 개
ARM = [i for i in range(3, 23) if i not in (14, 22)]


def main(paths: list[str]) -> int:
    runs = {}
    for p in sorted(paths):
        z = np.load(p, allow_pickle=True)
        a = np.asarray(z["actions"], dtype=np.float64)
        instr = str(z["instruction"])
        # 같은 문장을 두 번 돌린 것은 이름이 겹치므로 폴더 이름을 붙여 구분한다.
        key = f"{instr}  [{pathlib.Path(p).parent.name}]"
        runs[key] = a[:, ARM] if a.ndim == 2 else a.reshape(len(a), -1)[:, ARM]
        print(f"{key}\n    {p}  {a.shape}")

    n = min(len(v) for v in runs.values())
    print(f"\n공통 길이 {n} 스텝으로 잘라 비교한다\n")

    keys = list(runs)
    rows = []
    for x, y in itertools.combinations(keys, 2):
        a, b = runs[x][:n], runs[y][:n]
        diff = float(np.abs(a - b).mean())
        corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        same = x.split("  [")[0] == y.split("  [")[0]
        rows.append((same, x.split("  [")[0], y.split("  [")[0], diff, corr))

    base = [r[3] for r in rows if r[0]]
    if not base:
        print("🔴 같은 문장을 두 번 돌린 쌍이 없다 — 대조군 없이는 판정할 수 없다.")
        return 1
    ctrl = float(np.mean(base))

    # 같은 문장을 두 번 돌리면 같은 문장쌍이 여러 번 나온다. 그대로 나열하면
    # 표가 길어지고 어느 것이 대표값인지 알 수 없으므로 문장쌍으로 묶어 평균낸다.
    agg: dict = {}
    for same, x, y, diff, corr in rows:
        k = tuple(sorted((x, y)))
        agg.setdefault(k, {"same": same, "d": [], "c": []})
        agg[k]["d"].append(diff)
        agg[k]["c"].append(corr)

    print("=" * 96)
    print(f"{'비교':<58}{'차이':>10}{'대조군대비':>11}{'상관':>9}{'n':>5}")
    print("=" * 96)
    for (x, y), v in sorted(agg.items(), key=lambda kv: np.mean(kv[1]["d"])):
        d, c = float(np.mean(v["d"])), float(np.mean(v["c"]))
        label = f"{'[대조군] ' if v['same'] else ''}{x[:24]}  vs  {y[:24]}"
        print(f"{label:<58}{d:>10.3f}{d / ctrl:>10.2f}배{c:>9.3f}{len(v['d']):>5}")
    print("=" * 96)
    print("대조군(같은 문장 두 번)보다 뚜렷하게 크면 문장이 행동을 바꾼 것이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
