#!/usr/bin/env bash
# BEHAVIOR-1K (OmniGibson + Isaac Sim 5.1) 설치 스크립트
#
# 공식 setup.sh를 감싸서 다음을 한 번에 처리합니다.
#   1) conda가 없으면 Miniforge 설치 (conda-forge 기본, Anaconda ToS 불필요)
#   2) conda 환경 생성: python 3.11, pip, setuptools>=71,<81, wheel
#   3) --primitives(CuRobo)용 CUDA 12.8 toolkit을 conda 환경 안에 설치하고 activate/deactivate 훅 등록
#   4) BEHAVIOR-1K 저장소를 지정한 태그로 clone
#   5) 공식 setup.sh 실행 (OmniGibson, BDDL, JoyLo, eval, primitives, dataset)
#
# 라이선스: Isaac Sim EULA(--accept-nvidia-eula)와 BEHAVIOR Data Bundle EULA(--accept-dataset-tos)는
# 직접 읽고 동의한 경우에만 플래그를 붙이세요. 플래그가 없으면 setup.sh가 대화형으로 묻습니다.
#
# 사용 예:
#   bash setup/install_behavior.sh --dry-run
#   bash setup/install_behavior.sh --accept-nvidia-eula --accept-dataset-tos
#   bash setup/install_behavior.sh --no-dataset --no-primitives     # 최소 설치
# conda의 activate 스크립트가 미정의 변수를 참조하므로 set -u는 쓰지 않습니다.
set -eo pipefail

# ---------------- 기본값 ----------------
CONDA_DIR="${CONDA_DIR:-$HOME/miniforge3}"
ENV_NAME="behavior"
B1K_DIR="${B1K_DIR:-$HOME/BEHAVIOR-1K}"
B1K_TAG="v3.9.2"
B1K_REPO="https://github.com/StanfordVL/BEHAVIOR-1K.git"
CUDA_VERSION="12.8"
WITH_DATASET=true
WITH_PRIMITIVES=true
WITH_EVAL=true
WITH_JOYLO=true
ACCEPT_NVIDIA_EULA=false
ACCEPT_DATASET_TOS=false
DRY_RUN=false
SKIP_CHECKS=false

usage() {
    cat <<EOF
Usage: bash setup/install_behavior.sh [OPTIONS]

  --conda-dir DIR          conda 설치 위치 (기본: ~/miniforge3, 이미 conda가 있으면 그것을 사용)
  --env NAME               conda 환경 이름 (기본: behavior)
  --b1k-dir DIR            BEHAVIOR-1K clone 위치 (기본: ~/BEHAVIOR-1K)
  --tag TAG                BEHAVIOR-1K 태그 (기본: v3.9.2)
  --no-dataset             데이터셋 다운로드 생략 (나중에 setup/download_datasets.sh로 받을 수 있음)
  --no-primitives          CuRobo(모션 플래닝) 생략. CUDA toolkit 설치도 생략됨
  --no-eval                평가 의존성 생략 (eval은 joylo가 필요)
  --no-joylo               JoyLo(텔레오퍼레이션) 생략. --no-eval도 같이 적용됨
  --accept-nvidia-eula     NVIDIA Isaac Sim EULA에 동의
  --accept-dataset-tos     BEHAVIOR Data Bundle EULA에 동의 (비상업적 학술 연구 전용)
  --skip-checks            사전 점검(GPU, 디스크) 실패를 무시하고 진행
  --dry-run                실행할 명령만 출력
  -h, --help               도움말
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --conda-dir|--env|--b1k-dir|--tag)
            [[ $# -ge 2 && "$2" != -* ]] || { echo "$1 옵션에 값이 필요합니다"; usage; exit 1; } ;;&
        --conda-dir) CONDA_DIR="$2"; shift 2 ;;
        --env) ENV_NAME="$2"; shift 2 ;;
        --b1k-dir) B1K_DIR="$2"; shift 2 ;;
        --tag) B1K_TAG="$2"; shift 2 ;;
        --no-dataset) WITH_DATASET=false; shift ;;
        --no-primitives) WITH_PRIMITIVES=false; shift ;;
        --no-eval) WITH_EVAL=false; shift ;;
        --no-joylo) WITH_JOYLO=false; WITH_EVAL=false; shift ;;
        --accept-nvidia-eula) ACCEPT_NVIDIA_EULA=true; shift ;;
        --accept-dataset-tos) ACCEPT_DATASET_TOS=true; shift ;;
        --skip-checks) SKIP_CHECKS=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "알 수 없는 옵션: $1"; usage; exit 1 ;;
    esac
done

log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[경고] %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[오류] %s\033[0m\n' "$*" >&2; exit 1; }
run()  {
    if [[ "$DRY_RUN" == true ]]; then
        printf '  $'; printf ' %q' "$@"; printf '\n'
    else
        "$@"
    fi
}

# ---------------- 1. 사전 점검 ----------------
log "사전 점검"
[[ "$(uname -s)" == "Linux" ]] || die "Linux만 지원합니다."
ARCH="$(uname -m)"
[[ "$ARCH" == "x86_64" || "$ARCH" == "aarch64" ]] || die "지원하지 않는 아키텍처: $ARCH"
command -v git  >/dev/null || die "git이 필요합니다: sudo apt install git"
command -v curl >/dev/null || die "curl이 필요합니다: sudo apt install curl"

CHECK_FAILED=false
if ! command -v nvidia-smi >/dev/null; then
    warn "nvidia-smi가 없습니다. NVIDIA 드라이버(권장 580 이상)를 먼저 설치하세요."
    CHECK_FAILED=true
elif ! GPU_INFO="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1)"; then
    warn "nvidia-smi 실행 실패: ${GPU_INFO%%$'\n'*}"
    warn "드라이버를 설치한 직후라면 재부팅하세요 (Secure Boot이면 MOK 등록 필요)."
    CHECK_FAILED=true
else
    GPU_INFO="${GPU_INFO%%$'\n'*}"
    echo "  GPU: $GPU_INFO"
    # RT 코어가 없는 GPU(A100, H100 등)는 Isaac Sim이 지원하지 않습니다.
    if grep -qiE 'A100|A800|H100|H200|H800|A30\b|V100|P100' <<<"$GPU_INFO" \
       || ! grep -qiE 'RTX|Quadro|L40|L4\b|A10G?\b|A40|A6000' <<<"$GPU_INFO"; then
        warn "RT 코어가 있는 NVIDIA RTX급 GPU가 필요합니다 (A100/H100 등은 미지원)."
        CHECK_FAILED=true
    fi
fi

GLIBC_VER="$(ldd --version 2>&1 | head -1 | grep -oE '[0-9]+\.[0-9]+$' || true)"
echo "  GLIBC: ${GLIBC_VER:-unknown} (2.31 이상 필요, setup.sh가 2.31–2.34도 처리함)"

# 디스크: conda 환경 약 26 GB + 데이터셋 약 36 GB + 텍스처 캐시(씬마다 증가).
# 데이터셋 zip(29 GiB)은 TMPDIR에 받은 뒤 풀기 때문에, 같은 디스크라면 설치 중 최대 약 95 GB가 필요합니다.
df_dir="$(dirname "$B1K_DIR")"
while [[ ! -d "$df_dir" ]]; do df_dir="$(dirname "$df_dir")"; done   # dry-run에서도 디렉터리를 만들지 않음
tmp_dir="${TMPDIR:-/tmp}"
need_gb=45
if [[ "$WITH_DATASET" == true ]]; then
    need_gb=70
    [[ -d "$tmp_dir" && "$(stat -c %d "$tmp_dir")" == "$(stat -c %d "$df_dir")" ]] && need_gb=100
fi
free_gb=$(df -BG --output=avail "$df_dir" | tail -1 | tr -dc '0-9')
echo "  디스크 여유 ($df_dir): ${free_gb} GB (권장 ${need_gb} GB 이상)"
if (( free_gb < need_gb )); then
    warn "디스크 여유가 부족합니다."
    CHECK_FAILED=true
fi
if [[ "$WITH_DATASET" == true ]]; then
    # 데이터셋 zip(약 31.5 GB)은 TMPDIR(기본 /tmp)에 받은 뒤 압축을 풉니다.
    [[ -d "$tmp_dir" ]] || die "TMPDIR($tmp_dir)가 없습니다: mkdir -p $tmp_dir"
    tmp_free_gb=$(df -BG --output=avail "$tmp_dir" | tail -1 | tr -dc '0-9')
    echo "  임시 디렉터리 여유 ($tmp_dir): ${tmp_free_gb} GB (데이터셋 zip용 35 GB 이상 필요)"
    if (( tmp_free_gb < 35 )); then
        warn "TMPDIR 여유가 부족합니다. 'mkdir -p /큰/디스크/tmp && export TMPDIR=/큰/디스크/tmp' 후 다시 실행하세요."
        CHECK_FAILED=true
    fi
fi
if [[ "$CHECK_FAILED" == true && "$SKIP_CHECKS" == false ]]; then
    [[ "$DRY_RUN" == true ]] && warn "실제 실행에서는 사전 점검 실패로 중단됩니다 (무시하려면 --skip-checks)." \
        || die "사전 점검에 실패했습니다. 무시하려면 --skip-checks를 붙이세요."
fi

# ---------------- 2. conda (Miniforge) ----------------
log "conda 준비"
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    echo "  기존 conda 사용: $CONDA_BASE"
elif [[ -x "$CONDA_DIR/bin/conda" ]]; then
    CONDA_BASE="$CONDA_DIR"
    echo "  기존 Miniforge 사용: $CONDA_BASE"
else
    CONDA_BASE="$CONDA_DIR"
    installer="Miniforge3-Linux-${ARCH}.sh"
    echo "  Miniforge 설치: $CONDA_BASE"
    run curl -fsSL -o "/tmp/$installer" "https://github.com/conda-forge/miniforge/releases/latest/download/$installer"
    run bash "/tmp/$installer" -b -p "$CONDA_BASE"
    run rm -f "/tmp/$installer"
fi

if [[ "$DRY_RUN" == false ]]; then
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

# ---------------- 3. conda 환경 ----------------
log "conda 환경 '$ENV_NAME'"
if [[ -x "$CONDA_BASE/bin/conda" ]] && "$CONDA_BASE/bin/conda" env list | awk '{print $1}' | grep -x "$ENV_NAME" >/dev/null; then
    echo "  이미 있음. 그대로 사용합니다."
else
    run conda create -n "$ENV_NAME" python=3.11 pip "setuptools>=71,<81" wheel -c conda-forge --override-channels -y
fi
if [[ "$DRY_RUN" == false ]]; then
    conda activate "$ENV_NAME"
    [[ "${CONDA_DEFAULT_ENV:-}" == "$ENV_NAME" ]] || die "conda 환경 활성화 실패"
else
    printf '  $ conda activate %s\n' "$ENV_NAME"
fi

# ---------------- 4. CUDA toolkit (primitives/CuRobo용) ----------------
if [[ "$WITH_PRIMITIVES" == true ]]; then
    log "CUDA $CUDA_VERSION toolkit (conda 환경 안에 설치, 시스템 CUDA는 건드리지 않음)"
    # 훅이 CUDA_HOME=$CONDA_PREFIX로 설정하므로 conda 환경 안의 nvcc만 인정합니다(시스템 CUDA는 무시).
    if [[ "$DRY_RUN" == false ]]; then ENV_PREFIX="$CONDA_PREFIX"; else ENV_PREFIX="$CONDA_BASE/envs/$ENV_NAME"; fi
    if [[ -x "$ENV_PREFIX/bin/nvcc" ]] && "$ENV_PREFIX/bin/nvcc" -V | grep "release $CUDA_VERSION" >/dev/null; then
        echo "  nvcc $CUDA_VERSION 이미 있음: $ENV_PREFIX/bin/nvcc"
    else
        run conda install "cuda-compiler=$CUDA_VERSION" "cuda-toolkit=$CUDA_VERSION" -c conda-forge --override-channels -y
    fi
fi

# activate.d / deactivate.d 훅: CUDA 경로(CuRobo 빌드/JIT)와 Isaac Sim EULA 환경 변수
if [[ "$DRY_RUN" == false ]]; then
    HOOK_DIR="$CONDA_PREFIX/etc/conda"
else
    HOOK_DIR="<CONDA_PREFIX>/etc/conda"
fi
log "conda 활성화 훅 등록: $HOOK_DIR/{activate.d,deactivate.d}/zz_behavior.sh"
if [[ "$DRY_RUN" == false ]]; then
    if [[ "$ARCH" == "aarch64" ]]; then CUDA_TARGET="sbsa-linux"; else CUDA_TARGET="x86_64-linux"; fi
    mkdir -p "$HOOK_DIR/activate.d" "$HOOK_DIR/deactivate.d"
    # bash와 zsh 모두에서 동작하도록 eval만 사용합니다(${!var} 같은 bash 전용 문법 없음).
    {
        cat <<'EOF'
# generated by policy/setup/install_behavior.sh
for _behavior_v in CUDA_HOME CUDACXX CUDA_PATH CPATH LIBRARY_PATH LD_LIBRARY_PATH OMNI_KIT_ACCEPT_EULA; do
  eval "_BEHAVIOR_OLD_${_behavior_v}=\"\${${_behavior_v}-}\"; export _BEHAVIOR_OLD_${_behavior_v}"
done
unset _behavior_v
EOF
        if [[ "$WITH_PRIMITIVES" == true ]]; then
            cat <<EOF
# CUDA $CUDA_VERSION (conda-forge) for CuRobo build/JIT
export CUDA_HOME="\$CONDA_PREFIX"
export CUDACXX="\$CONDA_PREFIX/bin/nvcc"
export CUDA_PATH="\$CUDA_HOME"
export CPATH="\$CONDA_PREFIX/targets/$CUDA_TARGET/include\${CPATH:+:\$CPATH}"
export LIBRARY_PATH="\$CONDA_PREFIX/targets/$CUDA_TARGET/lib\${LIBRARY_PATH:+:\$LIBRARY_PATH}"
export LD_LIBRARY_PATH="\$CONDA_PREFIX/targets/$CUDA_TARGET/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
EOF
        fi
        if [[ "$ACCEPT_NVIDIA_EULA" == true ]]; then
            cat <<'EOF'
# NVIDIA Isaac Sim EULA accepted at install time (--accept-nvidia-eula)
export OMNI_KIT_ACCEPT_EULA=YES
EOF
        fi
    } > "$HOOK_DIR/activate.d/zz_behavior.sh"
    cat > "$HOOK_DIR/deactivate.d/zz_behavior.sh" <<'EOF'
# generated by policy/setup/install_behavior.sh
for _behavior_v in CUDA_HOME CUDACXX CUDA_PATH CPATH LIBRARY_PATH LD_LIBRARY_PATH OMNI_KIT_ACCEPT_EULA; do
  eval "_behavior_old=\"\${_BEHAVIOR_OLD_${_behavior_v}-}\""
  if [ -n "$_behavior_old" ]; then eval "export ${_behavior_v}=\"\$_behavior_old\""; else unset "$_behavior_v"; fi
  unset "_BEHAVIOR_OLD_${_behavior_v}"
done
unset _behavior_old _behavior_v
EOF
    # 훅을 현재 셸에 적용
    conda deactivate
    conda activate "$ENV_NAME"
fi

# ---------------- 5. BEHAVIOR-1K clone ----------------
log "BEHAVIOR-1K $B1K_TAG → $B1K_DIR"
if [[ -d "$B1K_DIR/.git" ]]; then
    cur_tag="$(git -C "$B1K_DIR" describe --tags --exact-match 2>/dev/null || echo unknown)"
    echo "  이미 있음 (현재: $cur_tag). clone을 건너뜁니다."
    [[ "$cur_tag" == "$B1K_TAG" ]] || warn "요청한 태그($B1K_TAG)와 다릅니다. 필요하면 직접 checkout 하세요."
else
    run git clone --depth 1 --branch "$B1K_TAG" "$B1K_REPO" "$B1K_DIR"
fi

# ---------------- 6. 공식 setup.sh ----------------
SETUP_ARGS=(--omnigibson --bddl)
[[ "$WITH_JOYLO" == true ]] && SETUP_ARGS+=(--joylo)
[[ "$WITH_EVAL" == true ]] && SETUP_ARGS+=(--eval)
[[ "$WITH_PRIMITIVES" == true ]] && SETUP_ARGS+=(--primitives)
[[ "$WITH_DATASET" == true ]] && SETUP_ARGS+=(--dataset)
[[ "$ACCEPT_NVIDIA_EULA" == true ]] && SETUP_ARGS+=(--accept-nvidia-eula)
[[ "$ACCEPT_DATASET_TOS" == true && "$WITH_DATASET" == true ]] && SETUP_ARGS+=(--accept-dataset-tos)

log "공식 setup.sh 실행: ./setup.sh ${SETUP_ARGS[*]}"
if [[ "$WITH_DATASET" == true ]]; then
    echo "  (PyTorch 등 pip 패키지 수 GB, Isaac Sim 휠 약 5 GB, 데이터셋 zip 약 32 GB를 받습니다. 네트워크에 따라 30분~수 시간)"
else
    echo "  (PyTorch 등 pip 패키지 수 GB, Isaac Sim 휠 약 5 GB를 받습니다)"
fi
[[ "$ACCEPT_NVIDIA_EULA" == true && ( "$WITH_DATASET" == false || "$ACCEPT_DATASET_TOS" == true ) ]] \
    || echo "  (--accept-* 플래그가 없으므로 곧 setup.sh가 약관을 보여 주고 y/N을 한 번 묻습니다)"
LOG_FILE="$B1K_DIR/setup-install.log"
if [[ "$DRY_RUN" == true ]]; then
    printf '  $ (cd %s && ./setup.sh %s 2>&1 | tee %s)\n' "$B1K_DIR" "${SETUP_ARGS[*]}" "$LOG_FILE"
elif ! (cd "$B1K_DIR" && ./setup.sh "${SETUP_ARGS[@]}" 2>&1 | tee "$LOG_FILE"); then
    die "공식 setup.sh가 실패했습니다. 로그: $LOG_FILE
  원인을 고친 뒤 같은 명령을 다시 실행하면 이어서 진행합니다 (conda 환경, Isaac Sim, 받은 데이터셋은 건너뜀)."
fi

log "완료"
cat <<EOF
다음으로 설치를 확인하세요 (새 셸이나 새 tmux 창에서도 같은 순서):
  cd $(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
  CONDA_DIR=$CONDA_BASE OG_ENV=$ENV_NAME source scripts/og_env.sh
  bash setup/check_env.sh --sim      # 첫 실행은 셰이더 컴파일로 수 분, 이후 약 30초
EOF
