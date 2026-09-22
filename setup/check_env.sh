#!/usr/bin/env bash
# OmniGibson 설치 점검. conda 환경을 활성화한 상태에서 실행합니다.
#   bash setup/check_env.sh          # 빠른 점검 (시뮬레이터를 띄우지 않음, 수 초)
#   bash setup/check_env.sh --sim    # 빈 씬을 헤드리스로 띄워 20 step 실행 (약 30초, 첫 실행은 셰이더 컴파일로 수 분)
set -o pipefail

RUN_SIM=false
[[ "${1:-}" == "--sim" ]] && RUN_SIM=true

pass=0; fail=0
ok()  { printf '  \033[32m[OK]\033[0m %s\n' "$*"; pass=$((pass+1)); }
bad() { printf '  \033[31m[FAIL]\033[0m %s\n' "$*"; fail=$((fail+1)); }
inf() { printf '  [..] %s\n' "$*"; }

echo "== GPU"
if command -v nvidia-smi >/dev/null; then
    ok "$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    inf "현재 GPU 메모리 사용: ${used} MiB (다른 시뮬이 돌고 있으면 VRAM이 부족할 수 있음)"
else
    bad "nvidia-smi 없음"
fi

echo "== Python / conda"
if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then ok "conda env: $CONDA_DEFAULT_ENV ($CONDA_PREFIX)"; else bad "conda 환경이 활성화되지 않음"; fi
pyver=$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)
[[ "$pyver" == "3.11" ]] && ok "python $pyver" || bad "python 3.11 필요 (현재: ${pyver:-없음})"
[[ "${OMNIGIBSON_HEADLESS:-}" =~ ^(1|true|True)$ ]] && ok "OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS}" \
    || inf "OMNIGIBSON_HEADLESS가 설정되지 않음 (SSH/디스플레이 없는 환경에서는 export OMNIGIBSON_HEADLESS=1 필요)"

echo "== 패키지"
if ! command -v python >/dev/null; then
    bad "python이 없습니다. 먼저 'source scripts/og_env.sh'로 conda 환경을 활성화하세요."
    echo; echo "결과: OK $pass / FAIL $fail"; exit 1
fi
if python - <<'EOF'
import importlib, os, sys
from importlib.metadata import version, PackageNotFoundError
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
failed = []
try:
    import torch
    cuda = torch.cuda.is_available()
    print(f"  torch {torch.__version__}, CUDA available={cuda}, device={torch.cuda.get_device_name(0) if cuda else '-'}")
    if not cuda:
        failed.append("torch.cuda")
except Exception as e:
    print(f"  torch: FAIL ({type(e).__name__}: {e})"); failed.append("torch")
try:
    print(f"  isaacsim {version('isaacsim')}")
except PackageNotFoundError:
    print("  isaacsim: FAIL (설치되지 않음)"); failed.append("isaacsim")
# curobo(--primitives), gello(--joylo), torch_cluster(--eval)는 선택 설치
for mod, required in [("omnigibson", True), ("bddl", True), ("curobo", False), ("gello", False), ("torch_cluster", False)]:
    try:
        m = importlib.import_module(mod)
        print(f"  {mod}: {getattr(m, '__version__', 'ok')}")
    except Exception as e:
        print(f"  {mod}: {'FAIL' if required else '없음 (선택)'} ({type(e).__name__}: {e})")
        if required:
            failed.append(mod)
sys.exit(1 if failed else 0)
EOF
then ok "필수 패키지 import"; else bad "필수 패키지 import 실패 (위 FAIL 참고)"; fi

echo "== 데이터셋"
DATA_PATH=$(python -c 'from omnigibson.macros import gm; print(gm.DATA_PATH)' 2>/dev/null | tail -1)
if [[ -n "$DATA_PATH" && -d "$DATA_PATH" ]]; then
    ok "DATA_PATH=$DATA_PATH"
    for d in behavior-1k-assets omnigibson-robot-assets 2026-challenge-task-instances; do
        if [[ -d "$DATA_PATH/$d" ]]; then
            v=$(cat "$DATA_PATH/$d/VERSION" 2>/dev/null || echo "-")
            ok "$d (version $v)"
        else
            bad "$d 없음 → bash setup/download_datasets.sh"
        fi
    done
    [[ -f "$DATA_PATH/omnigibson.key" ]] && ok "omnigibson.key 있음" || bad "omnigibson.key 없음 (데이터셋 라이선스 동의 필요)"
    n_scenes=$(ls "$DATA_PATH/behavior-1k-assets/scenes" 2>/dev/null | wc -l)
    inf "씬 ${n_scenes}개"
else
    bad "데이터 경로를 찾을 수 없음 (OMNIGIBSON_DATA_PATH 확인)"
fi

echo "== 디스크"
df -h "${DATA_PATH:-$HOME}" | tail -1 | awk '{print "  "$6": 사용 "$5", 여유 "$4}'
avail_gb=$(df -BG --output=avail "${DATA_PATH:-$HOME}" | tail -1 | tr -dc '0-9')
(( avail_gb >= 15 )) && ok "여유 ${avail_gb} GB" || bad "여유 ${avail_gb} GB (새 씬을 처음 로드하면 텍스처 캐시가 수 GB씩 늘어남)"

if [[ "$RUN_SIM" == true ]]; then
    echo "== 헤드리스 시뮬레이션 (빈 씬, 20 step)"
    inf "처음 실행하면 셰이더 컴파일로 수 분 걸릴 수 있습니다. 진행 로그: tail -f /tmp/og_check_env.$$.log"
    OMNIGIBSON_HEADLESS=1 python -u - <<'EOF' > /tmp/og_check_env.$$.log 2>&1
import os, time
os.environ["OMNIGIBSON_HEADLESS"] = "1"
import torch as th
import omnigibson as og
t0 = time.time()
env = og.Environment(configs={"scene": {"type": "Scene"}, "robots": [], "task": {"type": "DummyTask"}})
env.reset()
for _ in range(20):
    env.step(th.zeros(0))
print(f"@@SIM_OK load+20steps={time.time() - t0:.1f}s", flush=True)
og.shutdown()
EOF
    if grep -q "@@SIM_OK" /tmp/og_check_env.$$.log; then
        ok "$(grep -o '@@SIM_OK.*' /tmp/og_check_env.$$.log | sed 's/@@SIM_OK //')"
        rm -f /tmp/og_check_env.$$.log
    else
        bad "시뮬레이션 실패. 로그: /tmp/og_check_env.$$.log"
        grep -E "Traceback|Error|error" /tmp/og_check_env.$$.log | grep -v "py stderr" | tail -5 | sed 's/^/    /'
    fi
fi

echo
echo "결과: OK $pass / FAIL $fail"
(( fail == 0 ))
