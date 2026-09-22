#!/usr/bin/env bash
# BEHAVIOR 데이터셋 다운로드 (OmniGibson이 설치된 conda 환경에서 실행)
#
# 받는 것 (모두 Hugging Face: behavior-1k/zipped-datasets):
#   - omnigibson-robot-assets        로봇 에셋                        zip 약 0.6 GB → 풀면 약 2.8 GB
#   - behavior-1k-assets              씬 51개 + 객체 + 시스템           zip 약 31.5 GB → 풀면 약 33 GB
#   - 2026-challenge-task-instances   2026 챌린지 태스크 인스턴스        zip 약 0.1 GB → 풀면 약 0.6 GB
#   - omnigibson.key                  에셋 복호화 키 (라이선스 동의 시 자동 설치)
#
# 이미 있는 항목은 건너뜁니다. 저장 위치는 OMNIGIBSON_DATA_PATH (기본: <BEHAVIOR-1K>/datasets).
# zip은 TMPDIR(기본 /tmp)에 받은 뒤 압축을 풀고 지웁니다. 따라서 TMPDIR에 35 GB 이상 여유가 필요합니다.
#
# 사용 예:
#   bash setup/download_datasets.sh --accept-dataset-tos
#   TMPDIR=/data/tmp bash setup/download_datasets.sh --accept-dataset-tos
#   bash setup/download_datasets.sh --only robot,challenge   # 일부만
set -euo pipefail

ACCEPT=false
ONLY="robot,assets,challenge"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --accept-dataset-tos) ACCEPT=true; shift ;;
        --only) ONLY="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "알 수 없는 옵션: $1"; exit 1 ;;
    esac
done

# OMNIGIBSON_DATA_PATH가 아직 없으면 import omnigibson 자체가 AssertionError로 실패하므로 먼저 만듭니다.
[[ -n "${OMNIGIBSON_DATA_PATH:-}" ]] && mkdir -p "$OMNIGIBSON_DATA_PATH"
if ! err="$(python -c "import omnigibson" 2>&1 >/dev/null)"; then
    echo "[오류] omnigibson을 import할 수 없습니다:"; echo "$err" | tail -3
    echo "  - conda 환경 활성화: source scripts/og_env.sh"
    echo "  - OMNIGIBSON_DATA_PATH를 지정했다면 그 폴더가 있어야 합니다"
    exit 1
fi

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
DATA_PATH="$(python -c 'from omnigibson.macros import gm; print(gm.DATA_PATH)' 2>/dev/null | tail -1)"
echo "데이터 경로: $DATA_PATH"

if [[ ",$ONLY," == *",assets,"* ]]; then
    tmp_dir="${TMPDIR:-/tmp}"
    mkdir -p "$tmp_dir"
    tmp_free_gb=$(df -BG --output=avail "$tmp_dir" | tail -1 | tr -dc '0-9')
    data_free_gb=$(df -BG --output=avail "$DATA_PATH" | tail -1 | tr -dc '0-9')
    echo "여유 공간: TMPDIR($tmp_dir) ${tmp_free_gb} GB / 데이터 경로 ${data_free_gb} GB"
    if [[ ! -d "$DATA_PATH/behavior-1k-assets" ]]; then
        (( tmp_free_gb >= 35 )) || { echo "[오류] TMPDIR 여유가 35 GB 미만입니다. TMPDIR=/큰/디스크/tmp 로 지정하세요."; exit 1; }
        (( data_free_gb >= 40 )) || { echo "[오류] 데이터 경로 여유가 40 GB 미만입니다."; exit 1; }
        if [[ "$(stat -c %d "$tmp_dir")" == "$(stat -c %d "$DATA_PATH")" ]]; then
            (( data_free_gb >= 70 )) || { echo "[오류] TMPDIR와 데이터 경로가 같은 디스크입니다. 여유 70 GB 이상이 필요합니다."; exit 1; }
        fi
    fi
fi

py() { python -c "$1"; }

if [[ ",$ONLY," == *",robot,"* ]]; then
    echo "==> OmniGibson robot assets"
    py "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()"
fi
if [[ ",$ONLY," == *",assets,"* ]]; then
    echo "==> BEHAVIOR-1K assets (+ 복호화 키)"
    if [[ "$ACCEPT" == true ]]; then
        py "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=True)"
    else
        # 라이선스 전문을 출력하고 y/n을 묻습니다.
        py "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=False)"
    fi
fi
if [[ ",$ONLY," == *",challenge,"* ]]; then
    echo "==> 2026 BEHAVIOR Challenge task instances"
    py "from omnigibson.utils.asset_utils import download_2026_challenge_task_instances; download_2026_challenge_task_instances()"
fi

echo
echo "==> 설치된 데이터셋"
du -sh "$DATA_PATH"/*/ 2>/dev/null || true
[[ -f "$DATA_PATH/omnigibson.key" ]] && echo "복호화 키: 있음" || echo "복호화 키: 없음"
