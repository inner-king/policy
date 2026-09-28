"""32종 롤아웃 계획을 **데이터에서 직접 뽑는다.** CU 0, 시뮬레이터 불필요.

    uv run --project third_party/openpi python scripts/plan_rollout32.py
    uv run --project third_party/openpi python scripts/plan_rollout32.py --only-new

## 왜 필요한가

롤아웃 한 판에는 (태스크, 인스턴스, 데모 파일, 스킬 번호, 스킬 길이) 다섯 개가
있어야 한다. `rollout_all20.sh` 의 PLAN 은 이 다섯 개를 사람이 두 파일에서
옮겨 적어 만든 것이라, 지시문이 32종으로 늘면 같은 일을 열두 번 더 해야 한다.
다섯 개가 전부 `data/segments.parquet` 과 `data/meta/episodes/` 에 있으므로
여기서 뽑는다.

## 🔴 반드시 지켜야 하는 두 조건

1. **데모가 녹화된 인스턴스에서 돌린다** (HANDOFF 20.3). 인스턴스가 다르면 로봇
   시작 위치와 물건 배치가 달라 인계 자세가 0.002 -> 0.475 로 벌어진다.
2. **인계 스킬 앞에 재생 끊김이 없어야 한다** (HANDOFF 25.5). 우리 데이터셋은
   `move to` 세그먼트의 75% 를 버렸다(`build_dataset.py --nav-keep 0.25`). 재생이
   그 빈 구간을 지나면 **바퀴가 그 자리에 서서** 로봇이 엉뚱한 곳에서 인계받는다.
   팔·몸통은 절대 위치라 정확히 따라가므로 **인계 자세 차이로는 이 사고를 못
   잡는다** — 관절은 0.00x 로 맞는데 로봇은 딴 데 서 있다. 그래서 이미 내보낸
   데모(`outputs/demo/*.npz`)의 `gap_before` 표시를 읽어 끊김 0곳만 고른다.

## 읽는 법

- **✅ 바로 쓸 수 있다** — `outputs/demo/` 에 데모가 있고 그 스킬 앞에 끊김이 없다.
- **⬇️ 데모를 내보내야 한다** — 원본 parquet 을 받아 `export_demo_actions.py` 를
  돌려야 한다. 명령까지 찍어 준다.

⚠️ 스킬 길이는 **데모의 그 구간 길이**다. 정책에 줄 스텝 수가 아니다 —
정책 구간은 항상 2,700스텝(시뮬레이터 1분 30초)으로 잡는다 (HANDOFF 기준).
"""
import argparse
import glob
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "configs"))

import cooking_config as cc  # noqa: E402

DEMO_DIR = ROOT / "outputs" / "demo"


def load_existing_demos():
    """이미 내보낸 데모의 (스킬번호 -> 정보). 앞에 끊김이 있는지도 같이 센다."""
    out = {}
    for p in sorted(DEMO_DIR.glob("*.npz")):
        try:
            bd = json.loads(str(np.load(p, allow_pickle=True)["boundaries"]))
        except Exception as e:
            print(f"⚠️ {p.name} 을 못 읽었다 ({type(e).__name__})", file=sys.stderr)
            continue
        for b in bd:
            # 이 스킬까지 오는 동안 끊김이 몇 곳인가
            gaps = sum(1 for x in bd
                       if x.get("gap_before") and x["skill_idx"] <= b["skill_idx"])
            out.setdefault(b["instruction"], []).append(
                dict(demo=p.stem, skill=b["skill_idx"], n=b["n"], gaps=gaps))
    return out


def load_instance_ids():
    """episode_index -> task_instance_id. 원본 메타에만 있다."""
    files = sorted(glob.glob(str(ROOT / "data/meta/episodes/**/*.parquet"), recursive=True))
    if not files:
        return {}
    cols = ["episode_index", "task_instance_id"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    return dict(zip(df.episode_index, df.task_instance_id))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-new", action="store_true", help="신규 12종만 본다")
    a = ap.parse_args()

    seg_path = ROOT / "data/segments.parquet"
    if not seg_path.exists():
        sys.exit(f"세그먼트 표가 없다: {seg_path}  (scripts/build_segments.py 로 만든다)")
    seg = pd.read_parquet(seg_path, columns=[
        "task_name", "episode_index", "skill_idx", "instruction", "n_frames",
        "data/chunk_index", "data/file_index"])

    have = load_existing_demos()
    inst = load_instance_ids()
    targets = cc.NEW12 if a.only_new else cc.TOP32

    ready, need = [], []
    for ins in targets:
        # 1) 이미 내보낸 데모 중 끊김 0곳짜리
        clean = [d for d in have.get(ins, []) if d["gaps"] == 0]
        if clean:
            d = max(clean, key=lambda x: x["n"])       # 긴 구간이 판독하기 좋다
            ep = None
            for _, r in seg[seg.instruction == ins].iterrows():
                if f"{r.task_name}_ep{r.episode_index}" == d["demo"]:
                    ep = r
                    break
            ready.append((ins, d, ep))
            continue
        # 2) 새로 내보내야 한다 — 세그먼트가 긴 에피소드를 고른다
        sub = seg[seg.instruction == ins]
        if len(sub) == 0:
            need.append((ins, None))
            continue
        need.append((ins, sub.loc[sub.n_frames.idxmax()]))

    print(f"{'=' * 78}\n바로 쓸 수 있다 — {len(ready)}종\n{'=' * 78}")
    print("  이름                 태스크                인스턴스 데모                          스킬 길이")
    for ins, d, ep in ready:
        iid = inst.get(int(ep.episode_index), "?") if ep is not None else "?"
        task = ep.task_name if ep is not None else "?"
        new = "🆕" if ins in cc.NEW12 else "  "
        print(f"{new} {task:22} {str(iid):>6}  {d['demo']:34} {d['skill']:4} {d['n']:5}")
        print(f"     {ins}")

    print(f"\n{'=' * 78}\n데모를 내보내야 한다 — {len(need)}종\n{'=' * 78}")
    if need:
        dl, ex = set(), []
        for ins, r in need:
            if r is None:
                print(f"❌ {ins} — 세그먼트 표에 없다")
                continue
            iid = inst.get(int(r.episode_index), "?")
            new = "🆕" if ins in cc.NEW12 else "  "
            print(f"{new} {r.task_name:22} {str(iid):>6}  ep{int(r.episode_index):<8} "
                  f"스킬 {int(r.skill_idx):3} ({int(r.n_frames)}프레임)")
            print(f"     {ins}")
            dl.add(f"data/chunk-{int(r['data/chunk_index']):03d}/"
                   f"file-{int(r['data/file_index']):03d}.parquet")
            ex.append((r.task_name, int(r.episode_index)))

        print("\n# ── 1) 원본 액션 parquet 를 받는다 (하나 약 80MB) ──")
        print("third_party/openpi/.venv/bin/python - <<'PY'")
        print("from huggingface_hub import hf_hub_download")
        print(f"for f in {sorted(dl)}:")
        print('    hf_hub_download("behavior-1k/2026-challenge-demos", f,')
        print('                    repo_type="dataset", local_dir="data/raw_demo")')
        print("PY")
        print("\n# ── 2) 데모를 내보낸다 ──")
        for t, e in sorted(set(ex)):
            print(f"third_party/openpi/.venv/bin/python scripts/export_demo_actions.py "
                  f"--task {t} --episode {e}")
        print("\n# ── 3) 내보낸 데모에 '← 이동 세그먼트 빠짐' 이 안 찍히는지 확인하고")
        print("#       이 스크립트를 다시 돌린다.")

    print(f"\n합계: 바로 {len(ready)}종 / 내보내기 필요 {len(need)}종 (대상 {len(targets)}종)")
    print("⚠️ 정책 구간은 데모 길이가 아니라 **2,700스텝(시뮬레이터 1분 30초)** 으로 잡는다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
