#!/bin/bash
# 9차에서 **새로 배우는 12종**을 한 체크포인트로 한 번씩 돌린다.
#
#   COOKING_CKPT=checkpoints/pi05_cooking_top32_v5/run9/21999 \   # 생략하면 자동 선택
#   COOKING_CONFIG=pi05_cooking_top32_v5 \
#   OUT=outputs/run9_new12 bash scripts/rollout_new12.sh
#
# ## 이 파일이 왜 따로 있나
#
# `rollout_all20.sh` 는 기존 20종을 덮는다. 9차는 거기에 12종이 더해지므로,
# 평가는 **두 파일을 다 돌려야** 32종이 된다. 합치지 않고 나눠 둔 이유는 둘이
# 묻는 질문이 다르기 때문이다:
#
#   rollout_all20.sh  기존 20종이 **안 무너졌나** (망각 검사)
#   rollout_new12.sh  신규 12종을 **배웠나**     (이번 차수의 목적)
#
# 실행 조건(인스턴스·정책 2,700스텝·서버 매번 재기동·조기종료 끔)은
# `rollout_all20.sh` 와 **한 글자도 다르지 않다.** 조건이 갈리면 두 결과를 나란히
# 놓을 수 없다. 본체는 그 파일에서 그대로 가져왔다.
#
# ## ⚠️ 신규 12종은 누적 노출이 기존 20종의 10% 다
#
# 기존 20종은 2차부터 누적 25만 회를 넘겼는데 신규 12종은 0 에서 시작해 22,000회쯤
# 받는다 (HANDOFF 40.4). **여기서 성적이 낮은 것이 곧 "안 된다" 는 뜻은 아니다.**
# 볼 것은 성공률보다 **어디까지 갔는가**다:
#   1) 지시문이 가리키는 **그 물체** 쪽으로 갔나
#   2) 맞는 **동작**을 골랐나 (집으라는데 밀지 않았나)
#   3) 해냈나 (정밀도)
# 1 이 되면 "새 문장을 2만 회쯤만 보고도 물체를 지목할 수 있다" 는 뜻이다.
#
# ## 이 PLAN 은 어떻게 만들었나
#
# `scripts/plan_rollout32.py` 가 `data/segments.parquet` 과 `data/meta/episodes/`
# 에서 뽑았다. 사람이 옮겨 적지 않았다. 고른 기준 둘:
#   - **데모가 녹화된 인스턴스**에서 돌린다 (HANDOFF 20.3). 다르면 인계 자세가
#     0.002 -> 0.475 로 벌어진다.
#   - 인계 스킬 앞에 **재생 끊김이 0곳**인 것만 (HANDOFF 25.5). 우리 데이터셋은
#     `move to` 세그먼트의 75% 를 버려서, 재생이 빈 구간을 지나면 바퀴가 그 자리에
#     서고 로봇이 엉뚱한 곳에서 인계받는다. 팔·몸통은 절대 위치라 **인계 자세로는
#     이 사고를 못 잡는다.**
# 데모 5개는 이 PLAN 을 위해 09-24 에 새로 내보냈다 (chop_an_onion ep8468/8470,
# cook_a_frozen_pie ep14536, make_pizza ep9926, thawing_frozen_food ep14047).
# ⚠️ 데모는 `outputs/` 라 git 에 없다. 다른 기계에서는 `plan_rollout32.py` 가
# 찍어 주는 내려받기·내보내기 명령을 먼저 돌릴 것.
#
# ## 비용
#
# 추론 약 27분/건 x 12 = **약 5.5시간** + 재생. 재생 길이는 판마다 크게 다르다
# (board_pick_bar 252프레임 ~ board_pick_sink 13,443프레임). CU 0, 전부 로컬.
#
# ## 일부만 돌리기
#
#   ONLY="jar_lid_open tray_put_oven" OUT=outputs/run9_new12 bash scripts/rollout_new12.sh
#
# ## 🔬 대조군 — 정책에 안 묻고 같은 무대를 돌린다 (NULL_POLICY=1)
#
#   NULL_POLICY=1 OUT=outputs/null_new12 bash scripts/rollout_new12.sh
#
# 인계 자세를 굳힌 채(바퀴 0) 성공 판정만 똑같이 건다. 체크포인트와 상관없는
# 결과라 한 번 돌려 두면 모든 차수에 다시 쓴다. 판마다 약 10분(추론이 없다).
# 여기서 알 수 있는 것 셋:
#   1) 재생이 무대를 제대로 세우나 — README 첫머리의 "무대가 어긋났다" 경고
#   2) 판정기가 시뮬레이터에서 그 물체를 찾나 — "물체를 못 찾음" (HANDOFF 40.7-2)
#   3) 정책 없이도 성공으로 찍히나 — 찍히면 그 무대의 성공은 정책 덕이 아니다
# 성공하면 바로 멈춘다(--stop-on-success). 못 하면 2,700스텝을 다 채운다.
# 재생은 지금까지 같은 무대를 재현했다 (run7·run8 기존 20종 21판, 대조군·정책
# 신규 12종 12판 — 로그의 `[무대 검사]` 줄 기준). 그래도 본 실행의 로그에서
# 한 번 더 볼 것. ⚠️ README 의 "무대가 어긋났다" 머리 경고는 **부수 변화**(예:
# 놓기 과제에서 물체를 놓았다)에도 붙으므로 그것으로 세지 말 것 (HANDOFF 40.16).
#
# ⚠️ 이 스크립트를 다른 명령에서 이어 돌릴 때, 그 래퍼 명령줄에 정책 서버
# 프로세스 이름을 literal 로 넣지 말 것 — 여기서 쓰는 `pkill -f` 가 래퍼 자신을
# 죽인다 (종료코드 144).
set -u
ONLY=${ONLY:-}
STOP_ON_SUCCESS=${STOP_ON_SUCCESS:-0}     # 기본 끔 (과정을 끝까지 본다)
NULL_POLICY=${NULL_POLICY:-0}             # 1 이면 🔬 대조군 (머리말 참고)
cd "$(dirname "$0")/.."

# ⚠️ 스텝 번호를 박아 두지 않는다 — 스텝 수가 바뀌면(22,000 계획이지만 계산 셀이
# 그날 값으로 다시 낸다) 조용히 없는 경로를 가리키게 된다. 지정이 없으면 run 폴더의
# **가장 큰 스텝**을 고르고, 무엇을 골랐는지 반드시 찍는다.
RUN_DEFAULT=checkpoints/pi05_cooking_top32_v5/run10
if [ -n "${COOKING_CKPT:-}" ]; then
  CKPT=$COOKING_CKPT
else
  CKPT=$(ls -d "$RUN_DEFAULT"/[0-9]* 2>/dev/null | sort -t/ -k4 -n | tail -1)
  if [ -z "$CKPT" ]; then
    echo "❌ 체크포인트를 못 찾았다: $RUN_DEFAULT/<스텝>"
    echo "   COOKING_CKPT=<경로> 로 직접 지정할 것."
    exit 1
  fi
  echo "체크포인트를 자동으로 골랐다: $CKPT"
  echo "⚠️ 이것이 **열리는** 체크포인트인지 먼저 확인할 것 (8차는 마지막이 깨져 있었다):"
  echo "   third_party/openpi/.venv/bin/python scripts/finalize_checkpoint.py $RUN_DEFAULT"
fi
CONFIG=${COOKING_CONFIG:-pi05_cooking_top32_v5}
# 🔴 체크포인트가 설정의 폴더에 있는지 본다 — 짝이 안 맞아도 에러 없이 돈다 (HANDOFF 40.16).
case "$CKPT" in
  *"/$CONFIG/"*) ;;
  *) echo "❌ 체크포인트($CKPT)가 설정($CONFIG)의 폴더가 아니다."
     echo "   COOKING_CKPT 와 COOKING_CONFIG 를 같이 넘길 것."; exit 1 ;;
esac
echo "체크포인트 $CKPT / 설정 $CONFIG"
OUT=${OUT:-outputs/run10_new12}
STEPS=${STEPS:-2700}                      # 30fps 기준 90초
LOG=$OUT/logs
BEH=${BEH:-$HOME/miniconda3/envs/behavior/bin/python}
VENV=third_party/openpi/.venv/bin/python
mkdir -p "$LOG"

# 이름 태스크 인스턴스 데모 스킬 스킬길이  [|지시문덮어쓰기]
# 인스턴스는 그 데모의 task_instance_id. 재생이 짧은 것부터.
PLAN=(
  "jar_lid_open        can_meat               240 can_meat_ep956.npz                  6   195"   # 재생 3,178f  open the lid of the hinged jar
  "board_pick_sink     canning_food           173 canning_food_ep9719.npz            47   357"   # 재생 13,443f  pick up the cutting board from the drop in sink
  "board_back_counter  cook_cabbage           136 cook_cabbage_ep8294.npz            20   403"   # 재생 9,276f  put the cutting board back on the countertop
  "hotdog_put_micro    cook_hot_dogs           86 cook_hot_dogs_ep9080.npz            8   414"   # 재생 4,870f  put the hotdog in the microwave
  # 🔴 2026-09-26: 놓기 2종의 데모를 바꿨다 (HANDOFF 40.18). 9차 때의 ep8655(인스턴스 74)는
  #    재생이 피망을 **양손 다** 못 쥐었고(스킬 5) 비트는 오른손 것을 놓쳤다(스킬 12) —
  #    대조군과 run9 가 똑같이 어긋나서 두 판은 모델 성적이 아니었다. 이름에 2 를 붙인
  #    것은 옛 결과 폴더(null_new12/pepper_put_board 등)와 섞이지 않게 하려는 것이다.
  #    옛 두 줄:  pepper_put_board  74 ep8655 스킬 5 (431) / beet_put_board  74 ep8655 스킬 12 (785)
  #    새 데모 ep8652(인스턴스 70)는 기존 20종 pepper_chop·beet_chop 이 쓰는 것으로, run7·run9
  #    모두 그 재생 뒤 썰기에 성공했다 — 재생이 피망·비트를 도마까지 옮긴다는 간접 증거다.
  #    각 데모의 **첫 번째** 놓기(양손에 하나씩 쥔 채 시작)를 쓴다.
  "pepper_put_board2   slicing_vegetables      70 slicing_vegetables_ep8652.npz       5   253"   # 재생 2,405f  put the bell pepper on the cutting board
  "beet_put_board2     slicing_vegetables      70 slicing_vegetables_ep8652.npz      11   330"   # 재생 4,745f  put the beet on the cutting board
  "brisket_pick_fridge cook_a_brisket          13 cook_a_brisket_ep10207.npz          1   840"   # 재생 993f  pick up the brisket from the fridge
  "tray_put_oven       cook_a_frozen_pie      194 cook_a_frozen_pie_ep14536.npz      10  1500"   # 재생 8,070f  put the tray in the oven
  "knife_pick_bar      make_pizza             177 make_pizza_ep9926.npz               4  1680"   # 재생 918f  pick up the carving knife from the bar
  "parer_pick_bar      chop_an_onion          109 chop_an_onion_ep8468.npz            8  1937"   # 재생 3,396f  pick up the parer from the bar
  "plate_pick_fridge   thawing_frozen_food     77 thawing_frozen_food_ep14047.npz     2  2070"   # 재생 2,040f  pick up the plate from the fridge
  "board_pick_bar      chop_an_onion          112 chop_an_onion_ep8470.npz            1  2115"   # 재생 252f  pick up the cutting board from the bar
)

n_done=0; n_fail=0
for entry in "${PLAN[@]}"; do
  IFS='|' read -r fields prompt <<< "$entry"
  set -- $fields
  name=$1 task=$2 inst=$3 demo=$4 skill=$5 len=$6
  if [ -n "$ONLY" ] && [[ " $ONLY " != *" $name "* ]]; then continue; fi
  extra=$(( STEPS > len ? STEPS - len : 0 ))
  stoparg=(); [ "$STOP_ON_SUCCESS" = 1 ] && stoparg=(--stop-on-success)
  [ "$NULL_POLICY" = 1 ] && stoparg=(--null-policy --stop-on-success)
  promptarg=()
  [ -n "${prompt:-}" ] && promptarg=(--prompt "$prompt")

  if compgen -G "$OUT/$name/actions_*.npz" > /dev/null; then
    echo "════════ $name — 이미 있음, 건너뜀"; continue
  fi
  if [ ! -f "outputs/demo/$demo" ]; then
    echo "❌ $name — 데모가 없다: outputs/demo/$demo"
    echo "   scripts/export_demo_actions.py 로 먼저 만들 것"
    n_fail=$((n_fail+1)); continue
  fi
  # 재생이 버려진 `move to` 구간을 지나면 바퀴가 그 자리에 서서 로봇이 엉뚱한 곳에서
  # 인계받는다 (HANDOFF 25.5). 팔·몸통은 절대 위치라 **인계 자세로는 이 사고를 못 잡는다.**
  # export_demo_actions.py 가 남긴 gap_before 표시로 실행 전에 막는다.
  gaps=$($BEH -c "import json,sys,numpy as np; bd=json.loads(str(np.load(sys.argv[1],allow_pickle=True)['boundaries'])); print(' '.join(str(b['skill_idx']) for b in bd if b.get('gap_before') and b['skill_idx']<=int(sys.argv[2])))" "outputs/demo/$demo" "$skill")
  if [ -n "$gaps" ]; then
    echo "🔴 $name — 재생이 빠진 이동 구간을 지난다 (끊김 뒤 스킬: $gaps). 건너뛴다."
    echo "   원본 parquet 으로 데모를 다시 내보낼 것 (HANDOFF 25.5 / rollout_unseen.sh 머리말)."
    n_fail=$((n_fail+1)); continue
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

  echo "════════ $name  ($task inst$inst 스킬 $skill, 정책 $((len+extra))스텝 = $(( (len+extra)/30 ))초) ════════  $(date +%H:%M)"
  timeout 10800 $BEH scripts/rollout_b1k.py --task "$task" --instance "$inst" \
      --demo "outputs/demo/$demo" --skill "$skill" --extra-steps "$extra" \
      --output-dir "$OUT/$name" --config "$CONFIG" --save-actions --write-video "${stoparg[@]}" \
      "${promptarg[@]}" \
      > "$LOG/roll_$name.log" 2>&1
  echo "  종료코드 $?  $(date +%H:%M)"
  # 인계 자세를 반드시 본다 — 크면 재생이 무대를 재현하지 못한 것이라 그 판은
  # 모델의 실패가 아니다 (20.3 / 26장).
  grep -E "인계 자세|데모 재생|정책 구간|성공 판정|판정기|스킬 성공|이미 조건이 참" \
       "$LOG/roll_$name.log" | tail -8
  n_done=$((n_done+1))
done

pkill -f serve_policy_b1k
echo
echo "════════ 신규 12종 완료 — 성공 실행 $n_done / 실패 $n_fail ════════  $(date +%H:%M)"
echo
echo "== 한눈에 보기 =="
for entry in "${PLAN[@]}"; do
  IFS='|' read -r fields _ <<< "$entry"; set -- $fields; name=$1
  f="$LOG/roll_$name.log"
  [ -f "$f" ] || continue
  # 하네스의 마지막 줄: "성공: False (태스크 전체 기준) / 스킬 성공: True|False|None"
  ok=$(grep -oE "스킬 성공: [A-Za-z]+" "$f" | tail -1)
  judge=$(grep -m1 -oE "\[성공 판정\] [^(]*" "$f")
  # 인계 시점에 성공 조건이 이미 참이면 하네스가 판정을 끈다 = 재생이 물체 상태를 못 재현했다.
  # (예전에는 README 에만 있는 "무대가 어긋" 을 grep 해서 영영 안 걸렸고, grep -c 가 0 건일 때
  #  종료코드 1 로 끝나 `|| echo 0` 이 0 을 하나 더 붙이는 바람에 **모든 판에 🔴** 가 찍혔다.)
  stage=""
  grep -q "이미 조건이 참" "$f" && stage="   🔴 인계 때 이미 성공 상태 — 측정 실패, 모델 탓 아님"
  printf "  %-18s %-18s %s%s\n" "$name" "${ok:-결과줄없음(중단?)}" "$judge" "$stage"
done
echo
[ "$NULL_POLICY" = 1 ] && echo "🔬 대조군이다 — 정책에 묻지 않았다. 여기서 성공한 판은 **정책의 성과가 아니다.**"
echo "⚠️ 12종 전부 자동 판정기가 붙는 형태다(scripts/pick_top32.py 가 검사한다). 다만"
echo "   판정기가 **시뮬레이터에서 그 물체를 찾는지**는 이번이 처음이다. 로그에"
echo "   '물체를 못 찾음' 이 찍히면 그건 측정의 실패지 모델의 실패가 아니다."
echo "⚠️ 그리고 판정기보다 **영상이 우선이다** — run7 대 run8 은 자동 판정으로는"
echo "   run8 이 이기는데 영상으로 보면 졌다 (HANDOFF 40.1)."
