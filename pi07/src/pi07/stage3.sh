#!/usr/bin/env bash
# π0.7 재현 3단계 — 생성 서브골을 섞은 LL 추가 학습 (π0.7 VI-C). systemd 유닛으로 돌린다:
#
#   systemd-run --user --unit=pi07-stage3 --working-directory=$CAPSTONE \
#     -p Restart=on-failure -p RestartSec=300 -p MemoryMax=16G -p MemorySwapMax=4G \
#     -p StandardOutput=append:runs/train/pi07-stage3.log -p StandardError=append:runs/train/pi07-stage3.log \
#     bash src/pi07/stage3.sh
#
# 1) WM 실행 COMPLETED 대기 → 2) 마지막 WM 체크포인트로 서브골 N 장 생성 (GPU1, 재개 가능)
# 3) LL 실행 COMPLETED 대기 → 4) LL 최종 가중치에서 시작해 서브골 예시의 50% 를 생성 이미지로 바꿔 추가 학습 (GPU0)
# 어느 단계에서 죽어도 유닛이 다시 띄우면 끝난 단계는 건너뛴다.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

WM_RUN=${WM_RUN:-pi07wm-flux2kleinbase-6k}
LL_RUN=${LL_RUN:-pi07ll-pi05base-30k}
NEXT=${NEXT:-pi07ll-pi05base-30k-genmix-5k}
N_GEN=${N_GEN:-4000}
GEN_STEPS=${GEN_STEPS:-12}
GEN_DIR="$DATA/pi07/gen_subgoals/${WM_RUN}"
ts() { date -u '+%F %T'; }

last_ckpt() { ls -d "$RUNS/train/$1"/checkpoints/step-* 2>/dev/null | sort | tail -n 1; }

wait_done() {
  while [ ! -f "$RUNS/train/$1/COMPLETED" ]; do
    if systemctl --user is-failed --quiet "$1"; then echo "[$(ts)] $1 실패 — 중단"; exit 1; fi
    sleep 300
  done
}

if [ -f "$RUNS/train/$NEXT/COMPLETED" ]; then echo "[$(ts)] $NEXT 이미 완료"; exit 0; fi
if systemctl --user is-active --quiet "$NEXT"; then echo "[$(ts)] $NEXT 이미 실행 중"; exit 0; fi

echo "[$(ts)] 1) $WM_RUN 완료 대기"
wait_done "$WM_RUN"
WM_CKPT=$(last_ckpt "$WM_RUN")
if [ ! -f "$GEN_DIR/COMPLETED" ]; then
  echo "[$(ts)] 2) 서브골 생성 $N_GEN 장 ($WM_CKPT, ${GEN_STEPS} 스텝) → $GEN_DIR"
  sleep 30  # WM 프로세스가 GPU1 을 놓을 때까지
  CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PI07_FFMPEG_THREADS=1 \
    "$VLA_VENV/bin/python" -u "$SRC/pi07/gen_subgoals.py" --wm "$WM_CKPT" --out "$GEN_DIR" \
    --n "$N_GEN" --steps "$GEN_STEPS" --cfg 4.0 --bs 8
fi

echo "[$(ts)] 3) $LL_RUN 완료 대기"
wait_done "$LL_RUN"
LL_CKPT=$(last_ckpt "$LL_RUN")
echo "[$(ts)] 4) $NEXT 시작 (init $LL_CKPT)"
sleep 30
exec bash "$SRC/pi07/launch.sh" "$NEXT" 0 ll 5000 --init-from "$LL_CKPT" --gen-goal-dir "$GEN_DIR" --p-gen 0.5 \
  --lr 1e-5 --final-lr 2.5e-6 --warmup 200 --micro-batch 16 --accum 2 --goal-per-micro 4 --workers 5 \
  --save-every 1000 --val-every 1000 --milestone 5000
