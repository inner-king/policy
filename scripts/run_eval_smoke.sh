#!/usr/bin/env bash
# 2026 BEHAVIOR Challenge 공식 평가를 zero-action 정책(LocalPolicy)으로 돌려 파이프라인을 확인합니다.
#   bash scripts/run_eval_smoke.sh                                  # turning_on_radio, 300 step, 영상 저장
#   TASK=make_microwave_popcorn STEPS=100 OUT=/tmp/b1k bash scripts/run_eval_smoke.sh
#   FULLRES=1 bash scripts/run_eval_smoke.sh                         # 헤드 720² + 손목 480² RGB-D
set -eo pipefail
TASK="${TASK:-turning_on_radio}"
STEPS="${STEPS:-300}"
OUT="${OUT:-$HOME/b1k_eval}"
EXTRA=()
[[ "${FULLRES:-0}" == 1 ]] && EXTRA+=(--env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper)
[[ "${VIDEO:-1}" == 1 ]] && EXTRA+=(--write-video) || EXTRA+=(--no-write-video)

python -c "import omnigibson" >/dev/null 2>&1 || { echo "conda 환경을 먼저 활성화하세요 (source scripts/og_env.sh)"; exit 1; }
export OMNIGIBSON_HEADLESS=1
mkdir -p "$OUT"

python -u -m omnigibson.eval.eval \
    --task-name "$TASK" \
    --policy local \
    --max-steps "$STEPS" \
    --output-dir "$OUT" \
    "${EXTRA[@]}" 2>&1 | tee "$OUT/eval.log"

echo
echo "결과 JSON: $(ls -t "$OUT"/json/*.json 2>/dev/null | head -1)"
ls -t "$OUT"/videos/*.mp4 2>/dev/null | head -1 | sed 's/^/영상: /' || true
echo "(--policy local은 모든 행동이 0이므로 success=False, q_score=0.0이 정상입니다)"
