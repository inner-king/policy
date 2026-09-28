#!/usr/bin/env bash
# DM0.5 VLM LoRA 유무 마이크로벤치 (GPU 한 장). 결과 json/log 는 BENCH_OUT(기본 $RUNS/_bench/dm05-vlm-lora-ab) 에 쓴다.
#   [GPU=1] [EXTRA=--ddp TAG=_ddp] bash src/opendm/bench/run_bench.sh A:2 B:4 B_nohook:8
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BENCH_OUT="${BENCH_OUT:-$RUNS/_bench/dm05-vlm-lora-ab}"; mkdir -p "$BENCH_OUT"
export DM05_MODEL="${DM05_MODEL:-$MODELS/dm05}"
GPU=${GPU:-1}; EXTRA=${EXTRA:-}; TAG=${TAG:-}
for cfg in "$@"; do
  v=${cfg%%:*}; b=${cfg##*:}; f=$BENCH_OUT/${v}_bs${b}${TAG}
  nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 500 \
    "$OPENDM/.venv/bin/python" "$HERE/bench_vlm_lora_ab.py" --variant "$v" --bs "$b" --phys_gpu "$GPU" $EXTRA --out "$f.json" > "$f.log" 2>&1
  echo "$(basename "$f") exit=$? $(date +%T)"
done
