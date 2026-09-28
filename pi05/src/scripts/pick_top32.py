"""9차 학습 지시문 32종을 **데이터로 다시 검사한다.** CU 0, 시뮬레이터 불필요.

    HF_LEROBOT_HOME=$PWD/data/lerobot \
    uv run --project third_party/openpi python scripts/pick_top32.py

## 왜 스크립트로 두나

지시문 목록은 `configs/cooking_config.py` 에 문자열로 박혀 있다. 그 목록이
**맞는지**는 네 가지를 데이터·코드와 대조해야 아는데, 전부 사람이 눈으로
확인하면 놓친다. 실제로 이 검사가 계획의 오류를 하나 잡았다 — 넣으려던
`open/close the door of the electric refrigerator` 가 `rollout_unseen.sh` 의
CORE 6판 중 2판이라, 넣었으면 **유일한 일반화 측정이 사라질 뻔했다.**

## 무엇을 보나

1. **자동 판정기가 붙는 형태인가.** `rollout_b1k.py` 의 `make_success_check`
   정규식 5종(문 열기·닫기 / 켜기 / 집기 / 놓기 / 자르기) 중 하나에 걸려야 한다.
   안 걸리면 롤아웃이 돌아도 성패를 못 재고 영상 판독만 남는다.
   ⚠️ 여기서 보는 것은 **문장 형태**까지다. 판정기가 시뮬레이터에서 그 물체를
   실제로 찾는지는 롤아웃을 돌려야 안다 (HANDOFF 40.7-2).
2. 🔴 **미학습 기준선 9종을 건드리지 않는가.** `scripts/rollout_unseen.sh` 가
   일반화를 재는 데 쓰는 문장이다. 하나라도 학습에 넣으면 그 측정이 영영 사라진다.
3. **프레임이 상한(CAP) 이상인가.** 그 아래면 다른 지시문보다 덜 노출된다.
4. **토큰이 `max_token_len` 안에 들어가는가.** 넘으면 프롬프트가 **조용히 잘린다**
   — 에러가 안 난다.

## 돌려주는 것

종료 코드 0 이면 전부 통과, 1 이면 문제를 찾았다. 마지막에 노출 계산표도 찍는다.
"""
import collections
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "configs"))

import cooking_config as cc  # noqa: E402

# rollout_b1k.py 의 `make_success_check` 이 아는 형태. 순서까지 그대로 옮긴다
# (앞에서 걸리면 뒤는 안 본다).
JUDGE_FORMS = [
    ("문 열기·닫기", r"(open|close) the (?:door|lid) of the (.+)"),
    ("켜기",        r"turn on the (.+)"),
    ("집기",        r"pick up the (.+?)(?: from the .+)?$"),
    ("놓기",        r"put the (.+?) (on|in|into|onto|back on) the (.+?)$"),
    ("자르기",      r"chop the (.+?) with the (.+?)$"),
]

# 🔴 `scripts/rollout_unseen.sh` 가 쓰는 **미학습 기준선**. 학습에 넣으면 안 된다.
# 2026-09-24 에 `outputs/*unseen*/logs/roll_*.log` 의 "지시문:" 줄에서 그대로 읽었다.
# (para_* 4판은 배운 지시문을 다르게 말한 것이라 여기 없다 — 무대가 학습 안이다.)
UNSEEN_BASELINE = (
    "turn on the burner",
    "chop the head cabbage with the carving knife",
    "pick up the head cabbage from the fridge on the high shelf",
    "put the head cabbage on the cutting board",
    "open the right door of the top cabinet",
    "pick up the chili from the fridge on the high shelf",
    "close the door of the electric refrigerator",
    "open the door of the electric refrigerator",
    "pick up the tupperware from the electric refrigerator",
)


def judge_form(instruction: str) -> str | None:
    s = instruction.lower().strip()
    for name, pat in JUDGE_FORMS:
        if re.match(pat, s):
            return name
    return None


def frame_counts(repo_id: str):
    """지시문별 (프레임, 에피소드). `task_subset._frame_ranges` 와 같은 파일을 읽는다."""
    import os
    root = pathlib.Path(os.environ.get("HF_LEROBOT_HOME", ROOT / "data/lerobot")) / repo_id
    path = root / "meta/episodes.jsonl"
    if not path.exists():
        sys.exit(f"에피소드 목록이 없다: {path}\n"
                 "  HF_LEROBOT_HOME 을 데이터 위치로 지정할 것 "
                 "(예: HF_LEROBOT_HOME=$PWD/data/lerobot)")
    frames, eps = collections.Counter(), collections.Counter()
    for line in open(path):
        d = json.loads(line)
        t = (d.get("tasks") or [""])[0]
        frames[t] += d["length"]
        eps[t] += 1
    return frames, eps


def token_lengths(instructions):
    """state 25개가 전부 3자리인 **최악 조건**에서의 토큰 수.

    pi05 는 state 를 이산 언어 토큰으로 프리픽스에 넣으므로(`discrete_state_input`)
    지시문만 재면 안 된다.
    """
    import numpy as np
    from openpi.models import tokenizer as _tok
    t = _tok.PaligemmaTokenizer(cc.MAX_TOKEN_LEN_V2)
    worst = np.full(25, 999.0)
    return {ins: int(t.tokenize(ins, state=worst)[1].sum()) for ins in instructions}


def main() -> int:
    frames, eps = frame_counts(cc.REPO_ID)
    total = sum(frames.values())
    cap = cc.CAP_FRAMES_V5
    print(f"데이터: 지시문 {len(frames)}종 / 에피소드 {sum(eps.values()):,} / 프레임 {total:,}")
    print(f"검사 대상: TOP32 ({len(cc.TOP20)} + 신규 {len(cc.NEW12)}) / 상한 {cap:,}\n")

    try:
        toks = token_lengths(cc.TOP32)
    except Exception as e:                      # openpi 없이도 나머지 검사는 되게
        print(f"⚠️ 토큰 검사 건너뜀 ({type(e).__name__}: {e})")
        toks = {}

    ok = True
    print(f"{'':2} {'형태':12} {'프레임':>9} {'ep':>5} {'토큰':>5}  지시문")
    print("-" * 100)
    for ins in cc.TOP32:
        new = ins in cc.NEW12
        form = judge_form(ins)
        n = frames.get(ins, 0)
        tk = toks.get(ins)
        bad = []
        if n == 0:
            bad.append("데이터에 없음")
        if form is None:
            bad.append("판정기 없는 형태")
        if ins in UNSEEN_BASELINE:
            bad.append("🔴 미학습 기준선을 침범한다")
        if new and 0 < n < cap:
            bad.append(f"프레임이 상한 미만 ({n:,} < {cap:,})")
        if tk is not None and tk >= cc.MAX_TOKEN_LEN_V2:
            bad.append(f"토큰 {tk} >= 상한 {cc.MAX_TOKEN_LEN_V2} — 프롬프트가 잘린다")
        if bad:
            ok = False
        mark = "❌" if bad else ("🆕" if new else "✅")
        print(f"{mark} {form or '(없음)':12} {n:9,} {eps.get(ins,0):5,} "
              f"{tk if tk is not None else '-':>5}  {ins}")
        if bad:
            print(f"{'':21} ⚠️ " + " / ".join(bad))

    dup = [i for i, c in collections.Counter(cc.TOP32).items() if c > 1]
    if dup:
        ok = False
        print(f"\n❌ 중복된 지시문: {dup}")

    capped = sum(min(frames.get(i, 0), cap) for i in cc.TOP32)
    lo = min(min(frames.get(i, 0), cap) for i in cc.TOP32)
    hi = max(min(frames.get(i, 0), cap) for i in cc.TOP32)
    print(f"\n상한 적용 후 {capped:,} 프레임 / 지시문당 최대·최소 {hi:,}·{lo:,} ({hi/lo:.2f}배)")
    print("  ⚠️ 실제 학습에서는 이보다 조금 많다 (2026-09-24 실측 3,148,879). 상한은 에피소드")
    print("     **경계에서** 자르므로 마지막 에피소드 하나 길이만큼 넘칠 수 있다")
    print("     (`task_subset._subset` 의 `if acc >= cap: break`).")
    if toks:
        print(f"토큰 최대 {max(toks.values())} / 상한 {cc.MAX_TOKEN_LEN_V2}")

    # 배치 32 에 32종이라 지시문당 노출 = 스텝 수. run7 은 26,000스텝 20종이었다.
    run7 = 26_000 * 32 / len(cc.TOP20)
    print(f"\n노출 (배치 32) — run7 은 지시문당 {run7:,.0f}회였다")
    for steps in (20_000, cc.NUM_STEPS_V5, 32_000, 41_600):
        per = steps * 32 / len(cc.TOP32)
        mark = "  <- 설정값" if steps == cc.NUM_STEPS_V5 else ""
        print(f"  {steps:6,}스텝  지시문당 {per:8,.0f}회  run7 대비 {per/run7*100:5.1f}%"
              f"  {steps*3.0/3600:5.1f}시간(3.0초/스텝){mark}")

    print("\n" + ("✅ 전부 통과" if ok else "❌ 문제를 찾았다 — 위의 ⚠️ 줄을 볼 것"))
    print("⚠️ 이 검사는 **문장 형태**까지다. 판정기가 시뮬레이터에서 그 물체를 실제로")
    print("   찾는지는 롤아웃을 돌려야 안다 (HANDOFF 40.7-2).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
