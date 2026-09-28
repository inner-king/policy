#!/bin/bash
# 학습한 20종 전부를 한 체크포인트로 한 번씩 돌린다 — **학습 전 기준선**용.
#
#   COOKING_CKPT=checkpoints/pi05_cooking_top20_v3/run7/25999 \
#   OUT=outputs/run7_base20 bash scripts/rollout_all20.sh
#
# ## 왜 이 파일이 따로 있나
#
# 20종은 지금까지 `rollout_suite.sh`(9종) + `rollout_rest.sh`(11종) 두 파일에
# 나뉘어 있었다. 차수를 비교하려면 **같은 20종을 같은 조건으로** 돌려야 하는데,
# 두 파일은 기본 STEPS 가 1,800 과 2,700 으로 다르고 기본 체크포인트도 다르다.
# 그대로 이어 돌리면 "run7 이 run6 보다 나아졌나" 를 물을 때 조건이 섞인다.
#
# 그래서 **두 PLAN 을 그대로 합치고 조건만 하나로 통일**했다. 인스턴스·스킬
# 번호는 원본 두 파일에서 한 글자도 바꾸지 않았다 (아래 검증 참고).
#
# ## 검증 — 이 PLAN 이 TOP20 을 정확히 덮는가
#
# 2026-09-13 에 각 항목의 (데모, 스킬번호) 로 실제 지시문을 뽑아 대조했다.
#
#     PLAN 21개 (20종 + 문장만 바꾸는 A/B 1개)
#     TOP20 중 안 덮인 것  0개
#     PLAN 의 지시문 길이  전부 원본과 일치
#     재생이 지나는 끊김     21개 전부 0곳 (데모의 gap_before 표시로 확인)
#
# 다시 확인하려면:
#   python - <<'PY'
#   import json, numpy as np, re
#   src=open('configs/cooking_config.py').read()
#   top20=set(re.findall(r'"(.*?)"', re.search(r'TOP20 = \((.*?)\n\)', src, re.S).group(1)))
#   # PLAN 을 읽어 각 (demo, skill) 의 instruction 을 TOP20 과 대조한다
#   PY
#
# ## 조건 (두 파일의 결론을 합친 것)
#
# - **각 데모를 그 데모가 녹화된 인스턴스에서** 돌린다 (20.3). 인스턴스가 다르면
#   로봇 시작 위치와 물건 배치가 달라 인계 자세가 0.002 -> 0.475 로 벌어진다.
# - **바퀴를 묶지 않는다** (20.4). 학습한 20종 구간 자체에 이동이 들어 있다.
# - **정책 구간 2,700스텝 = 시뮬레이터 1분 30초.** 데모 길이에 맞춰 끊으면
#   "못 한 것" 과 "시간이 모자란 것" 이 구분되지 않는다.
# - **매 실행마다 정책 서버를 새로 띄운다** (18.5). 한 서버에서 이어 돌리면
#   난수열이 이어져 실행 간 비교가 오염된다.
# - **조기 종료는 기본 끔.** `Open` 판정이 관절 범위의 5% 라(32.2) 문이 살짝만
#   움직여도 성공으로 찍히고 **문을 확실히 여는 과정이 잘린다.** 기준선을 잴
#   때는 전 구간을 봐야 하므로 끈다. 켜려면 `STOP_ON_SUCCESS=1`.
#
# ## 자동 판정기는 20종 전부에 붙는다 (2026-09-07 이후)
#
# 🔴 **이 머리말은 한동안 "5종은 판정기가 없다" 고 적혀 있었다. 낡은 내용이었다.**
# `make_success_check` 은 커밋 4a51010(판정기 12/20 -> 20/20)에서 `put ...` 과
# `chop ...` 을 배웠다. 지금 아는 형태는 다섯이다:
#     (open|close) the (door|lid) of X   /  turn on X  /  pick up the X
#     put the X (on|in|into|onto|back on) the Y        /  chop the X with the Y
# 그러니 brussels_chop, beet_chop, pepper_chop, half_put, knife_back 도 자동
# 판정이 찍힌다. 로그 첫머리의 `[성공 판정]` 줄이 무엇으로 쟀는지 말해 준다.
#
# ⚠️ 그래도 **판정기보다 영상이 우선이다.** run7 대 run8 은 자동 판정으로는
# run8 이 11 대 8 로 이기는데, 영상으로 보면 run8 이 졌다 (HANDOFF 40.1).
# `outputs/video_review/notes.md` 가 그 판독 기록이다.
#
# ## ⚠️ `half_pick` 의 판정은 2026-09-07 에 고쳐졌다
#
# 예전에는 `half` 를 떼고 물체를 찾아 **통째 방울양배추를 쥐어도 성공**이었다
# (25.2). 지금은 반쪽을 먼저 찾는다. 그 이전 결과와 비교할 때 주의할 것.
#
# ## 비용
#
# 추론 약 169회 x 약 9.5초(CPU) = 약 27분/건 + 재생. 21건이면 **약 10~11시간**.
# Colab 학습이 도는 24시간 동안 로컬에서 병렬로 돌리기에 맞는 분량이다.
# CU 는 0 이다 — 전부 로컬이다.
#
# ## 일부만 돌리기
#
#   ONLY="fridge_open2 oven_open" OUT=outputs/run7_base20 bash scripts/rollout_all20.sh
#
# ⚠️ 이 스크립트를 다른 명령에서 이어 돌릴 때, 그 래퍼 명령줄에 정책 서버
# 프로세스 이름을 literal 로 넣지 말 것. 여기서 쓰는 `pkill -f` 가 래퍼 자신의
# 명령줄까지 매칭해 **자기를 죽인다** (2026-09-07 에 종료코드 144 로 죽었다).
set -u
ONLY=${ONLY:-}                            # 비어 있으면 전부 돌린다
STOP_ON_SUCCESS=${STOP_ON_SUCCESS:-0}     # 기본 끔 (문 열기 과정이 잘리는 것을 막는다)
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
OUT=${OUT:-outputs/run7_base20}
STEPS=${STEPS:-2700}                      # 30fps 기준 90초
LOG=$OUT/logs
BEH=${BEH:-$HOME/miniconda3/envs/behavior/bin/python}
VENV=third_party/openpi/.venv/bin/python
mkdir -p "$LOG"

# 이름 태스크 인스턴스 데모 스킬 스킬길이  [|지시문덮어쓰기]
# 인스턴스는 그 데모의 task_instance_id (data/meta/episodes). 재생이 짧은 것부터.
#
#   ── rollout_suite.sh 에서 온 9종 + A/B 1개 ──────────────────────────────
PLAN=(
  "jar_put            can_meat               240 can_meat_ep956.npz                 8   58"
  "half_put           cook_brussels_sprouts   71 cook_brussels_sprouts_ep19050.npz 20   55"
  "pepper_chop        slicing_vegetables      70 slicing_vegetables_ep8652.npz     29   63"
  "beet_chop          slicing_vegetables      70 slicing_vegetables_ep8652.npz     19   73"
  "half_pick          cook_brussels_sprouts   71 cook_brussels_sprouts_ep19050.npz 23   87"
  "brussels_pick      cook_brussels_sprouts   71 cook_brussels_sprouts_ep19050.npz 18  169"
  "knife_back         canning_food           173 canning_food_ep9719.npz           19  171"
  "oven_on            cook_brussels_sprouts   71 cook_brussels_sprouts_ep19050.npz 45  180"
  "knife_pick_counter canning_food           173 canning_food_ep9719.npz           15  189"
  "bratwurst          can_meat               240 can_meat_ep956.npz                 9  207"
  "brussels_chop      cook_brussels_sprouts   71 cook_brussels_sprouts_ep19050.npz 19  256"
  "board_pick         canning_food           288 canning_food_ep9793.npz            2  267"
  "micro_on           make_microwave_popcorn 100 make_microwave_popcorn_ep8093.npz  7  277"
  "oven_close         cook_brussels_sprouts   71 cook_brussels_sprouts_ep19050.npz 44  289"
  "micro_open         make_microwave_popcorn 250 make_microwave_popcorn_ep8154.npz  1  317"
  "knife_pick         canning_food           173 canning_food_ep9719.npz            0  471"
  "oven_open          cook_brussels_sprouts   71 cook_brussels_sprouts_ep19050.npz 39  510"
  "micro_close        make_microwave_popcorn 100 make_microwave_popcorn_ep8093.npz  6  543"
  "fridge_close       halve_an_egg           122 halve_an_egg_ep12484.npz           3  660"
  "fridge_open2       slicing_vegetables      74 slicing_vegetables_ep8655.npz      1 1610"
  # ── 같은 무대에서 문장만 바꾸는 A/B (21.4) ────────────────────────────────
  # micro_open 에서 정책은 'open the door of the microwave' 를 받고 **전자레인지를
  # 켰다.** 두 지시문 다 학습한 20종 안에 있다. 같은 무대에서 문장만 바꿔 돌려
  # 정책이 둘을 구분하는지 본다. 이 판은 TOP20 커버리지에 안 센다.
  "micro_on_ab        make_microwave_popcorn 250 make_microwave_popcorn_ep8154.npz  1  317|turn on the microwave"
)

n_done=0; n_fail=0
for entry in "${PLAN[@]}"; do
  IFS='|' read -r fields prompt <<< "$entry"
  set -- $fields
  name=$1 task=$2 inst=$3 demo=$4 skill=$5 len=$6
  if [ -n "$ONLY" ] && [[ " $ONLY " != *" $name "* ]]; then continue; fi
  extra=$(( STEPS > len ? STEPS - len : 0 ))
  stoparg=(); [ "$STOP_ON_SUCCESS" = 1 ] && stoparg=(--stop-on-success)
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
echo "════════ 20종 기준선 완료 — 성공 실행 $n_done / 실패 $n_fail ════════  $(date +%H:%M)"
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
echo "⚠️ 20종 전부 자동 판정기가 붙는다(2026-09-07 이후). 그래도 **영상이 우선이다** —"
echo "   run7 대 run8 은 판정으로는 run8 이 이기는데 영상으로는 졌다 (HANDOFF 40.1)."
echo "   각 폴더의 README.md 에 무엇을 해야 하는지가 시간 순서로 적혀 있다."
