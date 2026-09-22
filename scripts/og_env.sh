# shellcheck shell=bash
# OmniGibson 실행 환경 활성화. 실행하지 말고 source 하세요.
#   source scripts/og_env.sh            # 기본: ~/miniforge3, 환경 behavior
#   CONDA_DIR=/opt/conda OG_ENV=myenv source scripts/og_env.sh
# ~/.bashrc에 한 줄 추가해 두면 편합니다:
#   og_env() { source ~/policy/scripts/og_env.sh; }

_og_not_sourced=false
if [ -n "${ZSH_VERSION:-}" ]; then
    case "${ZSH_EVAL_CONTEXT:-}" in *:file*) ;; *) _og_not_sourced=true ;; esac
elif [ -n "${BASH_SOURCE:-}" ] && [ "${BASH_SOURCE[0]}" = "$0" ]; then
    _og_not_sourced=true
fi
if [ "$_og_not_sourced" = true ]; then
    echo "이 파일은 source로 실행해야 합니다: source $0" >&2
    exit 1
fi
unset _og_not_sourced

_og_conda_dir="${CONDA_DIR:-$HOME/miniforge3}"
if [ -f "$_og_conda_dir/etc/profile.d/conda.sh" ]; then
    . "$_og_conda_dir/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    . "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "conda를 찾을 수 없습니다 (CONDA_DIR=$_og_conda_dir)" >&2
    return 1
fi
conda activate "${OG_ENV:-behavior}" || return 1
export OMNIGIBSON_HEADLESS="${OMNIGIBSON_HEADLESS:-1}"   # SSH/디스플레이 없는 환경 기본값
unset _og_conda_dir
echo "conda env: $CONDA_DEFAULT_ENV, OMNIGIBSON_HEADLESS=$OMNIGIBSON_HEADLESS"
