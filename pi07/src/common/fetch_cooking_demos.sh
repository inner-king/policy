#!/usr/bin/env bash
# BEHAVIOR-1K 2026 챌린지 데모 중 "요리 관련" 태스크 17개를 HDD로 받는다.
#
# 태스크 선정 근거: bddl3/bddl/activity_definitions 1016개를 BDDL 목표 조건으로
# 훑어 요리 관련 225개를 뽑고(data/b1k/cooking_tasks.json), 그중 챌린지 100개
# (= 사람 데모가 있는 것)와 교집합을 낸 결과가 아래 17개다. 나머지 208개는 정의만 있다.
#
# 왜 HDD인가: 17개 전체(RGB+depth+parquet)가 501GB인데 NVMe(/)에는 254GB뿐이다.
# /data(22TB)에 원본을 두고 레포 안에는 심링크 뷰만 만든다.
#
# 청크 = 태스크. meta/info.json의 chunks_size=1000, 태스크당 에피소드 200개라
# chunk-NNN == task_index NNN 이고 episode_index 는 NNN*200 .. NNN*200+199 이다.
# annotations/ 만 경로가 chunk-NNN 이 아니라 task-0NNN 이라 include 패턴을 따로 준다.
#
# 재개 가능하다 — hf download 가 이미 받은 파일은 건너뛴다. 죽으면 그냥 다시 돌려라.
# ⚠️ 돌고 있는 동안 이 파일을 편집하지 마라. bash가 저장해둔 오프셋으로 되돌아가 읽어서
#    파일 길이가 바뀌면 남은 명령이 깨진다.
#
# 세션이 끊겨도 계속 돌게:
#   setsid nohup bash src/common/fetch_cooking_demos.sh >> /data/b1k/fetch.log 2>&1 < /dev/null &
#   RATE_MB=0 bash src/common/fetch_cooking_demos.sh      # 속도 제한 없이
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

REPO=behavior-1k/2026-challenge-demos
MIRROR="$B1K_HDD/2026-challenge-demos"   # HDD 원본 (LeRobot v3 루트 형태)
HF="$VLA_VENV/bin/hf"
WORKERS=16          # 8과 차이 없었다 — 무제한일 때 회선이 ~10 MB/s 로 병목이다
MAX_TRIES=20

RATE_MB=${RATE_MB:-5}   # 평균 목표 MB/s. 0이면 제한 없음.
IFACE=${IFACE:-enp16s0}
ON=5                    # 듀티 사이클 한 주기의 다운로드 구간(초)

# 기본 10초는 회선이 포화된 상태에서 너무 짧다. 어노테이션(작은 파일 다수) 받다가
# httpx.ReadTimeout 으로 죽는 걸 실제로 봤다.
export HF_HUB_DOWNLOAD_TIMEOUT=60

# chunk:task_name — 작은 것부터. 초반에 빨리 결과가 나오고 실패도 일찍 드러난다.
TASKS=(
  040:make_microwave_popcorn 042:chop_an_onion        062:halve_an_egg
  051:cook_a_brisket         046:cook_bacon           070:thawing_frozen_food
  045:cook_hot_dogs          096:cook_broccolini      072:cook_a_frozen_pie
  047:freeze_pies            050:freeze_fruit         004:can_meat
  041:cook_cabbage           043:slicing_vegetables   049:make_pizza
  048:canning_food           095:cook_brussels_sprouts
)

rx() { cat "/sys/class/net/$IFACE/statistics/rx_bytes"; }

# 속도 제한. tc 는 root 가 필요하고 trickle 은 설치돼 있지 않아서, 받는 프로세스를
# SIGSTOP/SIGCONT 로 듀티 사이클링해 평균을 맞춘다.
#
# ⚠️ 정지 중에도 커널 소켓 버퍼는 계속 수신한다. 그래서 "ON 구간 속도 × 듀티비"로
#    정지 시간을 계산하면 실제 평균이 목표를 한참 넘는다 (11 MB/s 그대로였다).
#    사이클 전체(ON+OFF)에서 실제로 흐른 바이트로 되먹임해야 맞는다.
#       new_off = (ON + off) * rate / target - ON
#    회선 속도가 변해도 몇 사이클 안에 목표로 수렴한다.
throttle() {
  local pid=$1 target=$((RATE_MB * 1000000)) off=0 cyc a b rate n=0
  while kill -0 "$pid" 2>/dev/null; do
    a=$(rx)
    kill -CONT "$pid" 2>/dev/null
    sleep "$ON"
    kill -STOP "$pid" 2>/dev/null
    [ "$off" -gt 0 ] && sleep "$off"
    b=$(rx)
    cyc=$((ON + off))
    rate=$(( (b - a) / cyc ))
    off=$(( cyc * rate / target - ON ))
    [ "$off" -lt 0 ] && off=0
    [ "$off" -gt 30 ] && off=30
    n=$((n + 1))
    if [ $((n % 12)) -eq 0 ]; then
      echo "    [rate] $((rate / 100000))e5 B/s  목표 $((target / 100000))e5  다음 정지 ${off}초"
    fi
  done
  kill -CONT "$pid" 2>/dev/null
}

# 스크립트가 죽을 때 자식을 STOP 상태로 남기지 않는다.
# INT/TERM 트랩은 반드시 exit 해야 한다 — 핸들러만 돌고 말면 kill 로 죽일 수 없는
# 스크립트가 된다 (실제로 그렇게 만들어서 kill -9 로 잡았다).
# 자식을 CONT 시킨 뒤 죽인다. STOP 상태로 두면 kill 이 안 먹고, 그냥 두면 재시도 루프가
# 만든 hf 가 고아(ppid=1)로 살아남아 제한 없이 계속 받는다 — 실제로 그렇게 됐었다.
cleanup() {
  pkill -CONT -P $$ 2>/dev/null || true
  pkill -P $$ 2>/dev/null || true
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

# 재시도 래퍼. 한 번 죽으면 set -e 가 12시간짜리 실행을 통째로 날린다.
dl() {
  local tries=0 pid tpid rc
  while true; do
    "$HF" download "$REPO" --repo-type dataset "$@" \
      --local-dir "$MIRROR" --max-workers "$WORKERS" &
    pid=$!
    tpid=""
    if [ "$RATE_MB" -gt 0 ]; then throttle "$pid" & tpid=$!; fi
    rc=0; wait "$pid" || rc=$?
    if [ -n "$tpid" ]; then kill "$tpid" 2>/dev/null || true; wait "$tpid" 2>/dev/null || true; fi
    [ "$rc" -eq 0 ] && return 0
    tries=$((tries + 1))
    if [ "$tries" -ge "$MAX_TRIES" ]; then
      echo "    ✗ ${MAX_TRIES}회 실패 — 중단"; return 1
    fi
    echo "    ⟳ 재시도 $tries/$MAX_TRIES (30초 후)  $(date -u +%H:%M:%S)Z"
    sleep 30
  done
}

mkdir -p "$MIRROR"
if [ "$RATE_MB" -gt 0 ]; then
  echo "속도 제한 ${RATE_MB} MB/s (SIGSTOP 듀티 사이클, $IFACE 실측 기준)"
else
  echo "속도 제한 없음"
fi

echo "=== [1/4] meta + 스킬 요약 받기 $(date -u +%F' '%H:%M:%S)Z ==="
# meta/ 는 100개 태스크 전체라도 6.4MB뿐이라 통째로 받는다.
dl --include "meta/*" --include "README.md" --include "LICENSE" --include "annotations/*.csv"

echo
echo "=== [2/4] 청크 ${#TASKS[@]}개 받기 ==="
i=0
for entry in "${TASKS[@]}"; do
  chunk=${entry%%:*}; name=${entry##*:}
  i=$((i + 1))
  echo "--- ($i/${#TASKS[@]}) chunk-$chunk  $name  $(date -u +%H:%M:%S)Z"
  dl --include "data/chunk-$chunk/*" \
     --include "videos/*/chunk-$chunk/*" \
     --include "annotations/task-0$chunk/*"
  echo "    누적 $(du -sh "$MIRROR" | cut -f1)  /  남은 디스크 $(df -h "$B1K_HDD" | awk 'NR==2{print $4}')"
done

echo
echo "=== [3/4] HF 파일 목록과 대조 ==="
"$VLA_VENV/bin/python" "$SRC/common/verify_b1k_mirror.py" || echo "⚠️ 검증 실패 — 스크립트를 다시 돌려라"

echo
echo "=== [4/4] 레포 심링크 뷰 만들기 → $DATA/b1k/ ==="
"$VLA_VENV/bin/python" "$SRC/common/make_task_views.py"

echo
echo "✅ 완료 $(date -u +%F' '%H:%M:%S)Z   원본 $MIRROR  ($(du -sh "$MIRROR" | cut -f1))"
