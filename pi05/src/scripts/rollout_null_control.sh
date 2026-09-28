#!/bin/bash
# 🔬 대조군: 정책에 묻지 않고 인계 자세를 굳힌 채 같은 무대를 돌린다.
# 그래도 성공하면 그 무대의 성공은 정책의 성과가 아니다.
set -u
cd "$(dirname "$0")/.."
BEH=${BEH:-$HOME/miniconda3/envs/behavior/bin/python}
VENV=third_party/openpi/.venv/bin/python
OUT=outputs/null_control; LOG=$OUT/logs; mkdir -p "$LOG"

# 정책 구간은 다른 스위트와 같은 90초(2,700스텝)로 맞춘다 (27.3 ④).
# 대조군은 길수록 보수적이다 — 더 오래 두고도 성공하지 않아야 "정책 덕분"이 선다.
STEPS=${STEPS:-2700}

PLAN=(
  "oven_open  cook_brussels_sprouts 71 cook_brussels_sprouts_ep19050.npz 39 510"
  "oven_on    cook_brussels_sprouts 71 cook_brussels_sprouts_ep19050.npz 45 180"
  "knife_pick_counter canning_food  173 canning_food_ep9719.npz          15 189"
)
for e in "${PLAN[@]}"; do
  set -- $e; name=$1 task=$2 inst=$3 demo=$4 skill=$5 len=$6
  extra=$(( STEPS > len ? STEPS - len : 0 ))
  rm -rf "$OUT/$name"
  pkill -f serve_policy_b1k; sleep 3
  $VENV scripts/serve_policy_b1k.py --device cpu --config pi05_cooking_top20_v3 \
    --ckpt checkpoints/pi05_cooking_top20_v3/run6/25000 > "$LOG/serve_$name.log" 2>&1 &
  for i in $(seq 1 90); do grep -q "대기 중 — ws://" "$LOG/serve_$name.log" && break; sleep 5; done
  echo "════════ 대조군 $name  (정책에 안 묻는다) ════════  $(date +%H:%M)"
  timeout 7200 $BEH scripts/rollout_b1k.py --task "$task" --instance "$inst" \
      --demo "outputs/demo/$demo" --skill "$skill" --extra-steps "$extra" \
      --output-dir "$OUT/$name" --save-actions --write-video --stop-on-success \
      --null-policy > "$LOG/roll_$name.log" 2>&1
  echo "  종료코드 $?  $(date +%H:%M)"
  grep -E "데모 재생|성공 판정|스킬 성공|성공:|대조군" "$LOG/roll_$name.log" | tail -6
done
pkill -f serve_policy_b1k
echo "════════ 대조군 완료 ════════  $(date +%H:%M)"
