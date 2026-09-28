#!/bin/bash
# 🔑 **학습하지 않은** 물체·기구를 정책이 다룰 수 있는가 — 일반화 기준선.
#
#   COOKING_CKPT=checkpoints/pi05_cooking_top20_v3/run7/25999 \
#   OUT=outputs/run7_unseen bash scripts/rollout_unseen.sh
#
# ## 왜 지금 재야 하나
#
# 다음 학습에서 **Gemma 2B 본체를 푸는 것**(v4)을 검토 중이다. 그러면 언어부가
# 학습한 20종 문장에 특화되어 **처음 보는 문장·물체를 다루는 능력이 떨어질 수
# 있다.** 그런데 지금 그 능력이 얼마인지 **한 번도 잰 적이 없다.** 학습 뒤에
# 재면 좋아졌는지 나빠졌는지 영영 알 수 없다. 그래서 학습 전에 잰다. CU 0.
#
# ## 무엇을 시험하나 (전부 학습 20종 밖의 지시문)
#
#   가까움: electric refrigerator 문 열기·닫기  <- 배운 fridge 와 물체·동작이 거의 같다
#   중간  : tupperware / head cabbage / chili 집기 <- 동작은 배웠고 물체가 새것
#   멂    : top cabinet **두 짝 문** 열기           <- 문 형태 자체가 다르다
#   EXTRA : head cabbage 옮기기·자르기, burner 켜기
#   PARA  : **배운 무대·배운 동작에 문장만 바꿔 말하기** ("shut the fridge")
#
# PARA 가 언어부 과적합을 가장 직접 잰다. 화면·물체·동작이 전부 학습 그대로이고
# 달라진 것은 표현뿐이라, 여기서 무너지면 "문장의 뜻" 이 아니라 "그 20개 문자열"
# 을 외운 것이다. 같은 무대의 원래 문장 결과는 rollout_all20.sh 가 낸다
# (all20 이름 = PARA 이름에서 para_ 를 뗀 것). 판정은 --judge-as 로 원래 문장에
# 맞춰 잰다. ⚠️ 판마다 1회라 우연 편차가 크다 — 성패 한 칸 차이로 결론내지 말 것.
#
# ## 🔴 재생이 끊기는 데모 — 실행 전에 막는다
#
# 우리 데이터셋은 `move to` 구간의 75%를 버렸다(nav-keep 0.25). 재생이 그 빈
# 구간을 지나면 **바퀴가 그 자리에 서서** 로봇이 엉뚱한 곳에서 인계받는다
# (HANDOFF 25.5). 팔·몸통은 절대 위치라 정확히 따라가므로 **인계 자세 차이로는
# 이 사고를 못 잡는다** — 관절은 0.00x 로 맞는데 로봇은 딴 데 서 있다.
#
# 그래서 `export_demo_actions.py` 가 데모에 남긴 `gap_before` 표시를 **실행 전에**
# 읽어, 인계 스킬 앞에 끊김이 있으면 그 판을 건너뛴다. 2026-09-14 확인:
#
#   데모                             gap_before 표시        이 스크립트의 판
#   cook_brussels_sprouts_ep19050   없음 (원본으로 메움)    CORE 3종 ✅
#   can_meat_ep956                  13, 17, 24             CORE 스킬 1 ✅
#   cook_cabbage_ep8294             없음 (09-14 원본으로 메움) CORE 1·2 ✅ / EXTRA 5·13 ✅
#   cook_a_brisket_ep10207          없음 (09-14 원본으로 메움) EXTRA 스킬 5 ✅
#
# 09-14 재내보내기 전후 비교: cook_cabbage 는 프레임 3,753 까지 액션이 완전히
# 같다(CORE 인계는 2,745 이전이라 영향 없음). 프레임 12,416 -> 13,256.
# cook_a_brisket 은 스킬 3 시작(2,523)부터 달라진다. 4,579 -> 5,359.
# 옛 파일은 되돌릴 일이 없어 보관하지 않는다.
#
# ## EXTRA 를 돌리려면 — 원본 parquet 두 개로 데모를 다시 내보낸다
#
# rollout_rest.sh 가 오븐 스킬에 쓴 방법과 같다(HANDOFF 25.5). 원본은 우리
# 데이터셋과 액션·state 가 비트 단위로 같다. 파일 하나 약 80MB.
#
#   third_party/openpi/.venv/bin/python - <<'PY'
#   from huggingface_hub import hf_hub_download
#   for f in ["data/chunk-041/file-006.parquet",    # cook_cabbage   ep8294
#             "data/chunk-051/file-000.parquet"]:   # cook_a_brisket ep10207
#       hf_hub_download("behavior-1k/2026-challenge-demos", f,
#                       repo_type="dataset", local_dir="data/raw_demo")
#   PY
#   third_party/openpi/.venv/bin/python scripts/export_demo_actions.py --task cook_cabbage   --episode 8294
#   third_party/openpi/.venv/bin/python scripts/export_demo_actions.py --task cook_a_brisket --episode 10207
#
# 다시 내보낸 데모에 "← 이동 세그먼트 빠짐" 이 안 찍히면 된다. 스킬 번호는
# 원본 번호라 바뀌지 않으므로 아래 PLAN 은 그대로 쓴다. 그 뒤 `TIER=all`.
#
# ## ⚠️ 자동 판정기가 붙는 것과 안 붙는 것
#
# `make_success_check` 이 아는 형태는 **다섯**이다 (2026-09-07 커밋 4a51010 이후):
# `(open|close) the (door|lid) of X` / `turn on X` / `pick up the X` /
# `put the X (on|in|into|onto|back on) the Y` / `chop the X with the Y`.
# 로그 첫머리의 `[성공 판정]` 줄이 무엇으로 쟀는지 말해 준다.
#
#   unseen_fridge_open/close, *_pick, unseen_burner_on   판정기 있음
#   unseen_cabbage_put / unseen_cabbage_chop             판정기 있음 (옛 주석은 틀렸다)
#   unseen_cabinet_open   ❌ 없다 — 'open the **right door** of' 가 정규식에 안 걸린다
#
# 🔴 `left|right door` 는 **일부러 안 고쳤다.** 정규식은 한 줄로 넓힐 수 있지만
# `Open` 상태가 물체 단위라 **왼쪽 문을 열어도 오른쪽 지시문이 성공으로 찍힌다.**
# 관절 단위(`open_fraction` 이 쓰는 `relevant_joints_info`)로 짜야 하는데 관절
# 이름에 left/right 가 들어 있는지 확인하지 못했다. 틀린 판정은 없는 판정보다
# 나쁘므로 확인 전까지는 그대로 둔다 (HANDOFF 40.7-4).
#
# 판정기가 물체를 못 찾으면 `[성공 판정] 물체를 못 찾음` 이 찍힌다. **그건 측정의
# 실패지 모델의 실패가 아니다.** 그때도 영상으로 본다.
#
# ## 읽는 법
#
# 성공/실패보다 **어디까지 갔는지**를 본다.
#   1) 물체를 맞게 골랐나  — 지시문이 가리키는 그 물체 쪽으로 갔나
#   2) 동작을 맞게 골랐나  — 열라고 했는데 켜지 않았나 (21.4 의 약점)
#   3) 해냈나              — 정밀도
# 1 이 되면 "언어로 물체를 지목하는 능력은 학습 밖에도 일반화된다" 는 뜻이고,
# 이것이 v4 전후로 어떻게 변하는지가 이번 측정의 핵심이다.
#
# ## 비용
#
# 약 27분/건 + 재생 (CPU 추론). CU 0.
#   TIER=core  CORE 6 + PARA 4 = 10건, 약 4.5시간
#   TIER=all   (기본) 여기에 EXTRA 3 = 13건, 약 6시간
#
# ⚠️ 다른 명령에서 이어 돌릴 때 그 명령줄에 정책 서버 프로세스 이름을 literal 로
# 넣지 말 것 — 아래 `pkill -f` 가 래퍼 자신을 죽인다 (종료코드 144).
set -u
ONLY=${ONLY:-}
TIER=${TIER:-all}                         # core(CORE+PARA) | all(+EXTRA)
STOP_ON_SUCCESS=${STOP_ON_SUCCESS:-0}     # 기본 끔 (과정을 끝까지 본다)
cd "$(dirname "$0")/.."

CKPT=${COOKING_CKPT:-checkpoints/pi05_cooking_top20_v3/run7/25999}
CONFIG=${COOKING_CONFIG:-pi05_cooking_top20_v3}
# 🔴 체크포인트가 설정의 폴더에 있는지 본다. 기본 체크포인트가 run7 이라
# `COOKING_CONFIG` 만 넘기면 **run7 을 다른 설정으로** 돌리면서도 에러가 안 난다
# (HANDOFF 40.12 의 옛 명령이 그랬다 — 40.16).
case "$CKPT" in
  *"/$CONFIG/"*) ;;
  *) echo "❌ 체크포인트($CKPT)가 설정($CONFIG)의 폴더가 아니다."
     echo "   COOKING_CKPT 와 COOKING_CONFIG 를 같이 넘길 것."; exit 1 ;;
esac
echo "체크포인트 $CKPT / 설정 $CONFIG"
OUT=${OUT:-outputs/run7_unseen}
STEPS=${STEPS:-2700}
LOG=$OUT/logs
BEH=${BEH:-$HOME/miniconda3/envs/behavior/bin/python}
VENV=third_party/openpi/.venv/bin/python
mkdir -p "$LOG"

# 이름 태스크 인스턴스 데모 스킬 스킬길이
# 인스턴스는 그 데모의 task_instance_id (data/meta/episodes, 2026-09-13 조회).
# 기존 PLAN 과 겹치는 7건으로 조회 방법을 검증했다 — 전부 일치했다.
CORE=(
  "unseen_tupper_pick  cook_brussels_sprouts  71 cook_brussels_sprouts_ep19050.npz  2 480"
  "unseen_fridge_close cook_brussels_sprouts  71 cook_brussels_sprouts_ep19050.npz  3 480"
  "unseen_cabbage_pick cook_cabbage          136 cook_cabbage_ep8294.npz            2 496"
  "unseen_chili_pick   cook_cabbage          136 cook_cabbage_ep8294.npz            1 545"
  "unseen_fridge_open  cook_brussels_sprouts  71 cook_brussels_sprouts_ep19050.npz  1 600"
  "unseen_cabinet_open can_meat              240 can_meat_ep956.npz                 1 894"
)
EXTRA=(
  "unseen_burner_on    cook_a_brisket         13 cook_a_brisket_ep10207.npz         5 270"
  "unseen_cabbage_chop cook_cabbage          136 cook_cabbage_ep8294.npz           13 301"
  "unseen_cabbage_put  cook_cabbage          136 cook_cabbage_ep8294.npz            5 501"
)
# 무대는 rollout_all20.sh 의 같은 이름 판과 똑같다. 뒤는 "정책에 줄 문장|판정 문장".
PARA=(
  "para_fridge_close   halve_an_egg          122 halve_an_egg_ep12484.npz           3 660|shut the fridge|close the door of the fridge"
  "para_micro_open     make_microwave_popcorn 250 make_microwave_popcorn_ep8154.npz 1 317|open the microwave|open the door of the microwave"
  "para_micro_on       make_microwave_popcorn 100 make_microwave_popcorn_ep8093.npz 7 277|switch on the microwave|turn on the microwave"
  "para_knife_pick     canning_food          173 canning_food_ep9719.npz            0 471|grab the knife off the cutting board|pick up the carving knife from the cutting board"
)

# PARA 를 먼저 — 가장 중요한 측정이고, --judge-as 가 도는지 30분 안에 확인된다.
PLAN=("${PARA[@]}" "${CORE[@]}")
[ "$TIER" = all ] && PLAN+=("${EXTRA[@]}")

n_done=0; n_fail=0; n_gap=0
for entry in "${PLAN[@]}"; do
  set -- ${entry%%|*}
  name=$1 task=$2 inst=$3 demo=$4 skill=$5 len=$6
  promptarg=()
  if [[ "$entry" == *"|"* ]]; then
    rest=${entry#*|}
    promptarg=(--prompt "${rest%%|*}" --judge-as "${rest#*|}")
  fi
  if [ -n "$ONLY" ] && [[ " $ONLY " != *" $name "* ]]; then continue; fi
  extra=$(( STEPS > len ? STEPS - len : 0 ))
  stoparg=(); [ "$STOP_ON_SUCCESS" = 1 ] && stoparg=(--stop-on-success)

  if compgen -G "$OUT/$name/actions_*.npz" > /dev/null; then
    echo "════════ $name — 이미 있음, 건너뜀"; continue
  fi
  if [ ! -f "outputs/demo/$demo" ]; then
    echo "❌ $name — 데모가 없다: outputs/demo/$demo"
    n_fail=$((n_fail+1)); continue
  fi
  gaps=$($BEH -c "import json,sys,numpy as np; bd=json.loads(str(np.load(sys.argv[1],allow_pickle=True)['boundaries'])); print(' '.join(str(b['skill_idx']) for b in bd if b.get('gap_before') and b['skill_idx']<=int(sys.argv[2])))" "outputs/demo/$demo" "$skill")
  if [ -n "$gaps" ]; then
    echo "🔴 $name — 재생이 빠진 이동 구간을 지난다 (끊김 뒤 스킬: $gaps). 건너뛴다."
    echo "   머리말의 'EXTRA 를 돌리려면' 대로 데모를 다시 내보낸 뒤 돌릴 것."
    n_gap=$((n_gap+1)); continue
  fi
  rm -rf "$OUT/$name"

  pkill -f serve_policy_b1k; sleep 3
  $VENV scripts/serve_policy_b1k.py --device cpu --config "$CONFIG" --ckpt "$CKPT" \
    > "$LOG/serve_$name.log" 2>&1 &
  for i in $(seq 1 90); do grep -q "대기 중 — ws://" "$LOG/serve_$name.log" && break; sleep 5; done
  if ! grep -q "대기 중 — ws://" "$LOG/serve_$name.log"; then
    echo "❌ $name — 정책 서버 기동 실패"; tail -5 "$LOG/serve_$name.log"
    n_fail=$((n_fail+1)); continue
  fi

  echo "════════ $name  ($task inst$inst 스킬 $skill, 정책 $((len+extra))스텝) ════════  $(date +%H:%M)"
  timeout 10800 $BEH scripts/rollout_b1k.py --task "$task" --instance "$inst" \
      --demo "outputs/demo/$demo" --skill "$skill" --extra-steps "$extra" \
      --output-dir "$OUT/$name" --config "$CONFIG" --save-actions --write-video "${stoparg[@]}" "${promptarg[@]}" \
      > "$LOG/roll_$name.log" 2>&1
  echo "  종료코드 $?  $(date +%H:%M)"
  grep -E "지시문:|판정은 정책|데모 재생|성공 판정|물체를 못 찾음|스킬 성공|이미 조건이 참" \
       "$LOG/roll_$name.log" | tail -6
  n_done=$((n_done+1))
done

pkill -f serve_policy_b1k
echo
echo "════════ 미학습 측정 완료 ($TIER) — 실행 $n_done / 실패 $n_fail / 끊김으로 건너뜀 $n_gap ════════  $(date +%H:%M)"
echo
echo "== 한눈에 보기 =="
for entry in "${PLAN[@]}"; do
  set -- $entry; name=$1
  f="$LOG/roll_$name.log"; [ -f "$f" ] || continue
  ok=$(grep -oE "스킬 성공: [A-Za-z]+" "$f" | tail -1)
  judge=$(grep -m1 -oE "\[성공 판정\] [^(]*" "$f")
  stage=""
  grep -q "이미 조건이 참" "$f" && stage="   🔴 인계 때 이미 성공 상태 — 측정 실패"
  printf "  %-20s %-18s %s%s\n" "$name" "${ok:-결과줄없음(중단?)}" "$judge" "$stage"
done
echo
echo "⚠️ 성공/실패보다 **어디까지 갔는지**를 볼 것. 각 폴더 README.md 에 모델이 무엇을"
echo "   해야 하는지가 인계 시점 0.0초 기준 시간 순서로 적혀 있다."
echo "⚠️ unseen_cabinet_open 만 판정기가 없다('right door of' 는 일부러 안 붙였다 — 머리말)."
