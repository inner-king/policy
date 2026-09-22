#!/usr/bin/env bash
# π0.7 재현 폐루프 평가 — OmniGibson(GPU0) + serve_pi07.py(GPU1). 학습이 두 GPU 를 쓰는 동안에는 못 돈다.
#
#   TASK=thawing_frozen_food LL=runs/train/pi07ll-pi05base-30k/checkpoints/step-030000 \
#   HL=runs/train/pi07hl-pi05base-8k/checkpoints/step-008000 \
#   [WM=runs/train/pi07wm-flux2kleinbase-6k/checkpoints/step-006000] [TAG=gc] \
#   [INSTANCES="0 1 2 3 4 5 6 7 8 9"] [MAX_STEPS=...] [META_CFG=1.7] [EXEC_STEPS=25] \
#     bash src/pi07/run_eval.sh
#
# HL 을 비우면 subtask 없이(태스크 문장만) LL 을 돈다 = π0.5 Fig.13 의 "implicit HL" 조건.
# WM 을 비우면 서브골 없이 돈다. MAX_STEPS 를 비우면 하네스 기본값(사람 평균 × 1.5)을 쓴다.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

: "${TASK:?TASK 필요}"; : "${LL:?LL 체크포인트 필요}"
HL=${HL:-}; WM=${WM:-}; TAG=${TAG:-}
INSTANCES=${INSTANCES:-"0 1 2 3 4 5 6 7 8 9"}
MAX_STEPS=${MAX_STEPS:-}
META_CFG=${META_CFG:-0}
EXEC_STEPS=${EXEC_STEPS:-25}
SIM_GPU=${SIM_GPU:-0}; SERVER_GPU=${SERVER_GPU:-1}; PORT=${PORT:-8000}

ll_run=$(basename "$(dirname "$(dirname "$LL")")"); ll_step=$(basename "$LL" | sed 's/step-0*//')
arm="ll"; [ -n "$HL" ] && arm="${arm}+hl"; [ -n "$WM" ] && arm="${arm}+wm"
ID="${ll_run}-at$((ll_step/1000))k-${TASK}-${arm//+/-}${TAG:+-$TAG}"
OUTDIR=$RUNS/eval/$ID
mkdir -p "$OUTDIR/logs"

SERVE_ARGS=(--ll "$LL" --exec-steps "$EXEC_STEPS" --meta-cfg "$META_CFG" --port "$PORT" --host 127.0.0.1
            --trace-dir "$OUTDIR/trace")
[ -n "$HL" ] && SERVE_ARGS+=(--hl "$HL")
[ -n "$WM" ] && SERVE_ARGS+=(--wm "$WM")
EVAL_ARGS=(--task-name "$TASK" --policy websocket --host 127.0.0.1 --port "$PORT" --mode public_test
           --instance-indices $INSTANCES --env-wrapper b1k_wrappers.RGBFullResWrapper --write-video --headless
           --output-dir "$OUTDIR")
[ -n "$MAX_STEPS" ] && EVAL_ARGS+=(--max-steps "$MAX_STEPS")

cat > "$OUTDIR/run.yaml" <<YAML
# 이 실행이 무엇이었는지의 단일 출처. 표·문서의 수치는 여기서 유도된다.
id: $ID
kind: eval
stack: pi07
task: $TASK
status: running
date: '$(date -u +%Y-%m-%d)'
policy:
  ll: $LL
  hl: ${HL:-none}
  wm: ${WM:-none}
serving:
  server: src/pi07/serve_pi07.py
  exec_steps: $EXEC_STEPS
  denoise_steps: 5
  hl_period_steps: 30
  wm_period_s: 4
  meta_cfg: $META_CFG
protocol:
  mode: public_test
  sim_gpu: $SIM_GPU
  server_gpu: $SERVER_GPU
  instances: "$INSTANCES"
  max_steps: ${MAX_STEPS:-harness default}
  env_wrapper: b1k_wrappers.RGBFullResWrapper
launcher: src/pi07/run_eval.sh
results: json/
YAML

echo "=== [1/2] π0.7 서버 (GPU $SERVER_GPU, :$PORT) $(date -u +%H:%M) ==="
CUDA_VISIBLE_DEVICES=$SERVER_GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$VLA_VENV/bin/python" "$SRC/pi07/serve_pi07.py" "${SERVE_ARGS[@]}" > "$OUTDIR/logs/serve.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT
ok=0
for i in $(seq 1 200); do
  sleep 3
  curl -s --max-time 3 http://127.0.0.1:$PORT/healthz >/dev/null && { ok=1; break; }
  kill -0 "$SERVER_PID" 2>/dev/null || break
done
[ "$ok" = 1 ] || { echo "서버 기동 실패:"; tail -20 "$OUTDIR/logs/serve.log"; exit 1; }

echo "=== [2/2] $TASK public_test [$INSTANCES] $(date -u +%H:%M) ==="
cd "$OG"
CUDA_VISIBLE_DEVICES=$SIM_GPU PYTHONUNBUFFERED=1 PYTHONPATH="$SRC_PYTHONPATH" \
OMNIGIBSON_DATA_PATH="$BEHAVIOR" OMNI_KIT_ACCEPT_EULA=YES \
B1K_ACTION_LOG_DIR="$OUTDIR/actions" \
"$OG_PY" -m omnigibson.eval.eval "${EVAL_ARGS[@]}" > "$OUTDIR/logs/eval.log" 2>&1 || echo "평가 비정상 종료"
sed -i 's/^status: running/status: done/' "$OUTDIR/run.yaml"
echo "=== 끝 $(date -u +%H:%M) → $OUTDIR ==="
