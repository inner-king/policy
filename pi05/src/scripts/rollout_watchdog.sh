#!/bin/bash
# 롤아웃 한 판이 **멈춘 채로** 시간 제한(3시간)까지 버티는 것을 막는다 (HANDOFF 40.16).
#
#   setsid nohup bash scripts/rollout_watchdog.sh >> outputs/<차수>_eval/watchdog.log 2>&1 < /dev/null &
#
# ⚠️ 연쇄 실행을 **먼저** 띄우고 이것을 띄운다 — 연쇄 실행이 없으면 바로 끝난다.
#
# 2026-09-25 05:24 에 대조군 `hotdog_put_micro` 가 2,330스텝에서 멈췄다. 메인 스레드
# 하나가 CPU 99% 로 돌면서 로그도 영상도 1시간 20분 동안 한 줄도 늘지 않았다.
# 롤아웃 스크립트의 제한은 판당 3시간이라, 그대로 두면 판마다 2시간 넘게 버린다.
#
# 규칙: 롤아웃 프로세스의 로그(표준출력 파일)가 **40분** 동안 안 바뀌면 끝낸다.
# 정상 판은 이렇게 오래 조용하지 않다 — 정책 구간은 추론마다(10~15초) 한 줄,
# 대조군은 300스텝마다 한 줄을 찍고, 가장 긴 재생(13,443프레임)도 판 전체가
# 17분이었다. 끝내면 롤아웃 스크립트가 종료코드 143 을 찍고 다음 판으로 넘어간다.
#
# 연쇄 실행(`run9_eval_chain.sh` 또는 `eval_chain.sh`)이 끝나면 이것도 스스로 끝난다.
set -u
cd "$(dirname "$0")/.."
LIMIT=${LIMIT:-2400}
echo "$(date '+%m-%d %H:%M') 감시 시작 — 로그가 ${LIMIT}초 멈추면 그 판을 끝낸다"
# 명령줄 **맨 앞**으로만 찾는다 — 연쇄 실행을 띄운 셸의 명령줄에도 같은 글자가 들어 있다.
while pgrep -f "^bash scripts/(run9_)?eval_chain.sh" > /dev/null; do
  for p in $(pgrep -f "^${BEH:-$HOME/miniconda3/envs/behavior/bin/python} scripts/rollout_b1k.py"); do
    log=$(readlink "/proc/$p/fd/1" 2>/dev/null) || continue
    [ -f "$log" ] || continue
    age=$(( $(date +%s) - $(stat -c %Y "$log") ))
    if [ "$age" -gt "$LIMIT" ]; then
      echo "$(date '+%m-%d %H:%M') 🔴 $(basename "$log") 가 ${age}초째 그대로 — pid $p 를 끝낸다"
      tail -2 "$log" | tr '\r' '\n' | tail -2 | sed 's/^/     마지막 줄: /'
      kill "$p"; sleep 30
      kill -0 "$p" 2>/dev/null && { kill -9 "$p"; echo "     TERM 로 안 끝나 KILL"; }
    fi
  done
  sleep 60
done
echo "$(date '+%m-%d %H:%M') 연쇄 실행이 끝나서 감시도 끝낸다"
