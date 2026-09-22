#!/usr/bin/env bash
# 경로 단일 출처. src/ 아래 모든 런처가 첫 줄에서 이걸 source 한다.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
#
# 절대경로를 스크립트마다 다시 적지 않는다 — 예전에 25개 파일에 흩어져 있었고
# 디렉토리를 한 번 옮길 때마다 전부 고쳐야 했다.

# 이 파일의 위치에서 저장소 루트를 유도한다 (하드코딩 금지)
CAPSTONE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- 우리 것 ---
SRC="$CAPSTONE/src"          # 우리가 쓴 코드 전부
RUNS="$CAPSTONE/runs"        # 실행 산출물 (train/ · eval/)
DATA="$CAPSTONE/data"        # 학습 데이터 뷰 (LeRobot v3)
MODELS="$CAPSTONE/models"    # 외부 배포 가중치 (공식 ckpt, init)
DOCS="$CAPSTONE/docs"        # engineering/ · analysis/

# --- 환경 루트 (경로 고정) ---
# venv 3개가 shebang에 절대경로를 박고 groot는 editable install까지 걸려 있다.
# 옮기면 Isaac 스택을 재설치해야 하므로 여기 있는 한 절대 움직이지 않는다.
# 다른 배치(예: ~/BEHAVIOR-1K + conda env)에서는 OG·OG_PY·VLA_VENV·BEHAVIOR 를 환경변수로 덮어쓴다.
GROOT="${GROOT:-$CAPSTONE/groot}"              # wensi-ai/Isaac-GR00T fork (behavior 브랜치) + .venv
OG="${OG:-$CAPSTONE/omnigibson}"            # BEHAVIOR-1K 챌린지 하네스 + .venv  ⚠️ 워킹트리 오염 금지
VLA_VENV="${VLA_VENV:-$CAPSTONE/vla/.venv}"       # LeRobot/SmolVLA 전용 venv (스크립트는 src/vla 에 있다)
BEHAVIOR="${BEHAVIOR:-$CAPSTONE/behavior}"        # OMNIGIBSON_DATA_PATH 가 가리키는 에셋·데모 루트
# 22TB HDD (/dev/sda1). 데모 mp4처럼 NVMe에 안 들어가는 대용량 원본이 여기 산다.
# 레포 안 data/b1k/<task>/ 는 여기를 가리키는 심링크 뷰다.
B1K_HDD="${B1K_HDD:-/data/b1k}"

# 평가 하네스 인터프리터. conda 환경이면 OG_PY=~/miniforge3/envs/behavior/bin/python 처럼 준다.
OG_PY="${OG_PY:-$OG/.venv/bin/python}"

# --- 파이썬 ---
# 평가 하네스가 --env-wrapper b1k_wrappers.RGBFullResWrapper 로 임포트한다.
SRC_PYTHONPATH="$SRC/common"

export CAPSTONE SRC RUNS DATA MODELS DOCS GROOT OG OG_PY VLA_VENV BEHAVIOR B1K_HDD SRC_PYTHONPATH
