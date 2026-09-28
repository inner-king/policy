#!/usr/bin/env bash
# OpenDM(dexmal/opendm) 전용 venv 를 dexmal/opendm:eval 이미지 구성에 맞춰 만든다.
#
#   bash src/opendm/setup_venv.sh              # $OPENDM/.venv 생성
#   VENV=/tmp/x/.venv bash src/opendm/setup_venv.sh   # 다른 위치에 시험 생성
#
# 이미지 구성: CUDA 12.8.1 · Python 3.10.20 · torch 2.11.0+cu128 · flash-attn 2.8.3 소스 빌드 ·
# pip install -e . (fast-infer extra 없음). 고정 버전은 image-pins.txt.
# 이미지 버전은 레지스트리 blob 의 site-packages 목록에서 뽑았다 (image-pins.txt 머리말).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

VENV="${VENV:-$OPENDM/.venv}"
# flash-attn 빌드에만 쓰는 툴킷. torch 가 cu128 이라 시스템 nvcc 13.1 로는 빌드가 거부된다(메이저 불일치).
CUDA128_HOME="${CUDA128_HOME:-$HOME/opt/cuda-12.8}"
# 이전에 빌드한 휠. 있으면 재컴파일(32코어·MAX_JOBS=6·sm80 만 22분 실측)을 건너뛴다.
FA_WHEEL="${FA_WHEEL:-$HOME/opt/wheels/cu128-torch2.11-sm80/flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl}"

if [[ -e "$VENV" ]]; then
    echo "이미 있다: $VENV — 지우고 다시 만들려면 직접 삭제할 것" >&2
    exit 1
fi

export VIRTUAL_ENV="$VENV"
uv venv "$VENV" --python 3.10.20

uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

if [[ -f "$FA_WHEEL" ]]; then
    uv pip install "$FA_WHEEL"
else
    # 3090/3090 Ti 는 sm86 이라 sm80 코드가 그대로 돈다. 이미지는 80;90;100;120 전부 빌드했다.
    uv pip install ninja packaging psutil wheel
    CUDA_HOME="$CUDA128_HOME" PATH="$CUDA128_HOME/bin:$VENV/bin:$PATH" \
    FLASH_ATTN_CUDA_ARCHS=80 FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=6 \
        uv pip install --no-build-isolation flash-attn==2.8.3
fi

uv pip install -e "$OPENDM" -r "$SRC/opendm/image-pins.txt"

"$VENV/bin/python" - <<'EOF'
import torch, transformers, flash_attn, opendm
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| flash_attn", flash_attn.__version__, "| cuda", torch.cuda.is_available())
EOF
