#!/usr/bin/env bash
# π0.7 재현 학습을 systemd 사용자 유닛으로 띄운다 — Claude 세션·SSH·로그아웃과 무관하게 돈다.
#
#   bash src/pi07/launch.sh <run-id> <gpu> <mode> <steps> [train.py 추가 인자...]
#   bash src/pi07/launch.sh pi07-hl-pi05base-10k 1 hl 10000
#   bash src/pi07/launch.sh pi07ll-... 0,1 ll 200000 --accum 1   # GPU 2장 데이터 병렬 (torchrun)
#
# 전제: loginctl enable-linger $USER (로그인 세션이 없어도 사용자 systemd 가 산다)
#
# 유닛 동작:
#   - Restart=on-failure: 크래시·커널 OOM 이면 2분 뒤 다시 뜨고 train.py 가 최신 체크포인트에서 재개한다.
#     2시간에 5번 넘게 죽으면 멈춘다 (StartLimit) — 같은 원인으로 무한 재시작하지 않도록.
#   - KillMode=mixed: 정지 시 SIGTERM 은 메인 프로세스에만 간다. 메인이 현재 스텝을 마치고 체크포인트를
#     쓰는 동안 데이터로더 워커가 살아 있어야 하기 때문이다.
#   - MemoryMax: cgroup 상한. 넘으면 이 유닛 안에서만 OOM 이 난다 (시스템 전체가 스왑으로 멈추지 않게).
#
# 조회·정지:
#   systemctl --user status <run-id>
#   tail -f runs/train/<run-id>/logs/train.log
#   systemctl --user stop <run-id>          # 체크포인트 쓰고 끝남. 다시 launch 하면 이어서 돈다
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

RUN_ID="$1"; GPU="$2"; MODE="$3"; STEPS="$4"; shift 4
MEM_MAX="${PI07_MEM_MAX:-18G}"
SWAP_MAX="${PI07_SWAP_MAX:-8G}"
RUN_DIR="$RUNS/train/$RUN_ID"
mkdir -p "$RUN_DIR/logs"

if systemctl --user is-active --quiet "$RUN_ID"; then
  echo "이미 돌고 있음: $RUN_ID"; exit 1
fi
systemctl --user reset-failed "$RUN_ID" 2>/dev/null || true
rm -f "$RUN_DIR/STOP"
# run.yaml 상태 갱신 (queued → running). 완료 시 train.py 가 done 으로 바꾼다
[ -f "$RUN_DIR/run.yaml" ] && sed -i -E 's/^status: (queued|interrupted)\b/status: running/' "$RUN_DIR/run.yaml"

PY="$VLA_VENV/bin/python"
# GPU 인자에 쉼표가 있으면 데이터 병렬 (torchrun). 유효 배치 = micro-batch × accum × GPU 수.
NPROC="$(echo "$GPU" | awk -F, '{print NF}')"
if [ "$NPROC" -gt 1 ]; then
  CMD=("$PY" -u -m torch.distributed.run --nproc_per_node="$NPROC" --master_port="${PI07_MASTER_PORT:-29517}"
       "$SRC/pi07/train.py" --mode "$MODE" --run-id "$RUN_ID" --steps "$STEPS" "$@")
else
  CMD=("$PY" -u "$SRC/pi07/train.py" --mode "$MODE" --run-id "$RUN_ID" --steps "$STEPS" "$@")
fi
echo "[$(date -u '+%F %T')] launch gpu=$GPU mem_max=$MEM_MAX cmd: ${CMD[*]}" >> "$RUN_DIR/logs/launcher.log"

systemd-run --user --unit="$RUN_ID" \
  --working-directory="$CAPSTONE" \
  -p Restart=on-failure -p RestartSec=120 \
  -p StartLimitIntervalSec=7200 -p StartLimitBurst=5 \
  -p KillMode=mixed -p TimeoutStopSec=600 \
  -p MemoryMax="$MEM_MAX" -p MemorySwapMax="$SWAP_MAX" \
  -p StandardOutput=append:"$RUN_DIR/logs/train.log" \
  -p StandardError=append:"$RUN_DIR/logs/train.log" \
  --setenv=CUDA_VISIBLE_DEVICES="$GPU" \
  --setenv=PYTHONUNBUFFERED=1 \
  --setenv=OMP_NUM_THREADS=4 \
  --setenv=MKL_NUM_THREADS=4 \
  --setenv=TOKENIZERS_PARALLELISM=false \
  --setenv=PI07_FFMPEG_THREADS=1 \
  --setenv=PI07_MAX_OPEN_VIDEOS=6 \
  --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${CMD[@]}"

sleep 2
systemctl --user status "$RUN_ID" --no-pager | head -8 || true
