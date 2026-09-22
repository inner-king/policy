#!/usr/bin/env bash
# 이미 떠 있는 π0.7 서버(serve_pi07.py, 포트별 조건)에 붙어 이 머신에서 시뮬레이터 평가를 돈다.
# 조건은 포트가 정한다 (예: 8010 = HL 켬, 8011 = 플랫). 여러 조건을 순서대로 돌리려면 CONDS 에 "포트:run-id" 를 나열.
#
#   TASK=thawing_frozen_food HOST=<서버 주소> INSTANCES="0 1 2 3 4" \
#   CONDS="8010:pi07ll-...-ll-hl 8011:pi07ll-...-ll" SIM_GPU=0 bash src/pi07/run_eval_client.sh
#
# run-id 마다 run.yaml 을 쓴다 (없을 때만). 체크포인트는 서버 쪽 인자라 여기서는 알 수 없으므로
# 기록용으로 LL= HL= WM= 을 넘기면 그대로 적고, 안 넘기면 unknown 으로 남는다.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
: "${TASK:?}"; : "${CONDS:?}"; HOST=${HOST:?정책 서버 주소 필요}; INSTANCES=${INSTANCES:-"0 1 2 3 4 5 6 7 8 9"}
SIM_GPU=${SIM_GPU:-0}; MAX_STEPS=${MAX_STEPS:-}
for cond in $CONDS; do
  PORT=${cond%%:*}; RID=${cond#*:}
  OUT=$RUNS/eval/$RID; mkdir -p "$OUT/logs"
  until curl -s --max-time 3 "http://$HOST:$PORT/healthz" >/dev/null; do echo "서버 대기 $HOST:$PORT"; sleep 30; done
  echo "=== $RID  $TASK [$INSTANCES] :$PORT  $(date -u '+%F %T') ==="
  ARGS=(--task-name "$TASK" --policy websocket --host "$HOST" --port "$PORT" --mode public_test
        --instance-indices $INSTANCES --env-wrapper b1k_wrappers.RGBFullResWrapper --write-video --headless
        --output-dir "$OUT")
  [ -n "$MAX_STEPS" ] && ARGS+=(--max-steps "$MAX_STEPS")
  # 같은 run-id 를 나눠 돌리는 경우(인스턴스 분할, super2 병행)에 손으로 채운 run.yaml 을 덮어쓰지 않는다
  if [ -e "$OUT/run.yaml" ]; then
    echo "run.yaml 이 이미 있어 그대로 둔다: $OUT/run.yaml"
  else
    cat > "$OUT/run.yaml" <<YAML
# 이 실행이 무엇이었는지의 단일 출처. 표·문서의 수치는 여기서 유도된다.
id: $RID
kind: eval
stack: pi07
task: $TASK
status: running
date: '$(date -u +%Y-%m-%d)'
policy:                          # 서버 기동 인자라 클라이언트는 모른다 — LL/HL/WM 환경변수로 넘긴 값을 그대로 적는다
  ll: ${LL:-unknown}
  hl: ${HL:-unknown}             # 이 포트에서 HL 을 쓰는지는 서버의 --port / --flat-port 가 정한다
  wm: ${WM:-unknown}
serving:
  server: src/pi07/serve_pi07.py (이미 떠 있는 서버 :$PORT — exec_steps 등 서빙 인자는 서버 기동 기록 참조)
protocol:
  mode: public_test
  sim_gpu: $SIM_GPU
  instances: "$INSTANCES"
  max_steps: ${MAX_STEPS:-harness default}
  env_wrapper: b1k_wrappers.RGBFullResWrapper
launcher: src/pi07/run_eval_client.sh
results: json/
YAML
  fi
  (cd "$OG" && CUDA_VISIBLE_DEVICES=$SIM_GPU PYTHONUNBUFFERED=1 PYTHONPATH="$SRC_PYTHONPATH" \
     OMNIGIBSON_DATA_PATH="$BEHAVIOR" OMNI_KIT_ACCEPT_EULA=YES B1K_ACTION_LOG_DIR="$OUT/actions-local" \
     "$OG_PY" -m omnigibson.eval.eval "${ARGS[@]}") >> "$OUT/logs/eval-local.log" 2>&1 \
     || echo "평가 비정상 종료: $RID"
  sed -i 's/^status: running/status: done/' "$OUT/run.yaml"
  echo "=== 끝 $RID $(date -u '+%F %T') ==="
done
