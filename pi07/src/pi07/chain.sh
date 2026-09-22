#!/usr/bin/env bash
# 앞 실행이 끝나면 다음 실행을 띄운다 — systemd 유닛으로 돌려 세션과 무관하게 이어지게 한다.
#
#   bash src/pi07/chain.sh <앞 run-id> <다음 run-id> <gpu> <mode> <steps> [train.py 인자...]
#
# 예: HL 이 끝나면 같은 GPU 에서 월드 모델 학습
#   systemd-run --user --unit=pi07-chain-wm --collect \
#     bash src/pi07/chain.sh pi07hl-pi05base-8k pi07wm-flux2kleinbase-8k 1 wm 8000 --micro-batch 8 ...
#
# 앞 실행의 COMPLETED 파일을 5분마다 본다. 앞 유닛이 실패 상태로 멈췄으면(StartLimit 초과 등)
# 다음 실행을 띄우지 않고 끝낸다 — 사람이 원인을 본 뒤 다시 걸어야 한다.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
PREV="$1"; NEXT="$2"; shift 2
PREV_DIR="$RUNS/train/$PREV"
echo "[$(date -u '+%F %T')] 대기: $PREV → $NEXT $*"
while [ ! -f "$PREV_DIR/COMPLETED" ]; do
  if systemctl --user is-failed --quiet "$PREV"; then
    echo "[$(date -u '+%F %T')] $PREV 유닛 실패 — $NEXT 를 띄우지 않는다"; exit 1
  fi
  # 유닛이 없는데 COMPLETED 도 없으면 사람이 멈춘 것이다 (systemctl stop) — 다시 띄울 때까지 계속 기다린다.
  sleep 300
done
# 앞 프로세스가 GPU 를 완전히 놓을 때까지
sleep 30
echo "[$(date -u '+%F %T')] $PREV 완료 → $NEXT 시작"
exec bash "$SRC/pi07/launch.sh" "$NEXT" "$@"
