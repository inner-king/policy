#!/bin/bash
# 원격 평가 스트림 — 시뮬레이터만 다른 머신에서 돌리고 정책 서버(serve_pi07.py)에 websocket 으로 붙는다.
# $E 에 src/pi07/eval_remote.py 와 src/common/b1k_wrappers.py 사본을 둔다 (PYTHONPATH=$E).
# 조건은 포트가 정한다 (8010 = HL 켬, 8011 = 플랫). 결과는 $E/<run-id>/ 에 쓰고 학습 머신 쪽 runs/eval/ 로 가져간다.
# 설치 계정과 실행 계정이 다를 때만 EXP_PATH·OMNIGIBSON_APPDATA_PATH 로 Isaac 앱·캐시 경로를 쓰기 가능한 곳에 둔다.
# 기록된 평가(인스턴스 5-9)는 이 경우였다: PY·OMNIGIBSON_DATA_PATH 는 설치 계정의 conda env 와 데이터셋 경로,
# EXP_PATH=$E/isaac_shadow/apps (isaacsim/apps 의 파일마다 심링크 + exts·extscache 등 형제 폴더),
# OMNIGIBSON_APPDATA_PATH=$E/og_appdata 로 주고 돌렸다. isaacsim 은 EXP_PATH 가 이미 있으면 그 값을 쓴다.
TASK=${TASK:-thawing_frozen_food}; HOST=${HOST:?정책 서버 주소 필요}; INSTANCES=${INSTANCES:-"5 6 7 8 9"}
CONDS=${CONDS:-"8010:pi07ll-pi05base-30k-at30k-thawing_frozen_food-ll-hl 8011:pi07ll-pi05base-30k-at30k-thawing_frozen_food-ll"}
E=${E:-$HOME/pi07_eval}
PY=${PY:-$HOME/miniforge3/envs/behavior/bin/python}
[ -n "${EXP_PATH:-}" ] && export EXP_PATH
[ -n "${OMNIGIBSON_APPDATA_PATH:-}" ] && export OMNIGIBSON_APPDATA_PATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  OMNIGIBSON_DATA_PATH=${OMNIGIBSON_DATA_PATH:-$HOME/BEHAVIOR-1K/datasets} OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PYTHONPATH=$E
for cond in $CONDS; do
  PORT=${cond%%:*}; RID=${cond#*:}; OUT=$E/$RID; mkdir -p $OUT/logs
  until curl -s --max-time 5 http://$HOST:$PORT/healthz >/dev/null; do echo "server wait $HOST:$PORT"; sleep 30; done
  echo "=== $RID $TASK [$INSTANCES] :$PORT $(date -u '+%F %T') ==="
  cd $E && B1K_ACTION_LOG_DIR=$OUT/actions-super2 $PY $E/eval_remote.py --task-name $TASK --policy websocket \
    --host $HOST --port $PORT --mode public_test --instance-indices $INSTANCES \
    --env-wrapper b1k_wrappers.RGBFullResWrapper --write-video --headless --output-dir $OUT >> $OUT/logs/eval-super2.log 2>&1 \
    || echo "eval failed: $RID"
  echo "=== done $RID $(date -u '+%F %T') ==="
done
