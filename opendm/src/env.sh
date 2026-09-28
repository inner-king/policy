#!/usr/bin/env bash
# 경로 단일 출처. src/ 아래 모든 런처가 첫 줄에서 이걸 source 한다.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
#
# 기본값은 이 파일 위치에서 유도한 실험 루트(opendm/) 아래 배치다. 다른 배치에서는 환경변수로 덮어쓴다.

# 이 파일의 위치에서 실험 루트를 유도한다 (하드코딩 금지)
CAPSTONE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- 우리 것 ---
SRC="$CAPSTONE/src"          # 우리가 쓴 코드 전부
RUNS="$CAPSTONE/runs"        # 실행 산출물 (train/ · eval/)
DATA="$CAPSTONE/data"        # 학습 데이터 뷰 (LeRobot v3 등)
MODELS="$CAPSTONE/models"    # 외부 배포 가중치 (models/dm05 = HF Dexmal/DM05)

# --- 환경 루트 ---
# OPENDM: dexmal/opendm 클론 + .venv (src/opendm/setup_venv.sh 로 생성). .venv 가 editable install 이라 옮기면 다시 만든다.
OPENDM="${OPENDM:-$CAPSTONE/opendm}"
OG="${OG:-$CAPSTONE/omnigibson}"                # BEHAVIOR-1K 챌린지 하네스 + .venv
VLA_VENV="${VLA_VENV:-$CAPSTONE/vla/.venv}"     # 브리지·개루프 프로브가 쓰는 venv (websockets·msgpack·PIL·lerobot)
BEHAVIOR="${BEHAVIOR:-$CAPSTONE/behavior}"      # OMNIGIBSON_DATA_PATH 가 가리키는 에셋·데모 루트
B1K_HDD="${B1K_HDD:-/data/b1k}"                 # 데모 원본 미러 (대용량 HDD)

# 평가 하네스 인터프리터. conda 환경이면 OG_PY=~/miniforge3/envs/behavior/bin/python 처럼 준다.
OG_PY="${OG_PY:-$OG/.venv/bin/python}"

# --- 파이썬 ---
# 평가 하네스가 --env-wrapper b1k_wrappers.RGBFullResWrapper 로 임포트한다.
SRC_PYTHONPATH="$SRC/common"

export CAPSTONE SRC RUNS DATA MODELS OPENDM OG OG_PY VLA_VENV BEHAVIOR B1K_HDD SRC_PYTHONPATH
