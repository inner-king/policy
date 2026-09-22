#!/usr/bin/env bash
# 평가가 다 끝나면 공유 π0.7 서버(pi07-serve)를 내린다 — GPU0 을 비워 두어야 3단계 LL 학습이 뜬다.
# 끝남 판정: 로컬 평가 유닛이 멈췄고, 평가 실행들의 trace 파일이 IDLE_MIN 분 동안 갱신되지 않았다
# (원격 super2 스트림도 HL 질의마다 trace 를 쓰므로 원격 종료까지 잡힌다).
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
IDLE_MIN=${IDLE_MIN:-30}
UNITS=${UNITS:-pi07-eval-local}
while true; do
  busy=0
  for u in $UNITS; do systemctl --user is-active --quiet "$u" && busy=1; done
  if [ $busy -eq 0 ]; then
    newest=$(find "$RUNS"/eval/pi07ll-*/trace -name '*.jsonl' -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
    now=$(date +%s)
    if [ -n "$newest" ] && [ $(( now - ${newest%.*} )) -gt $(( IDLE_MIN * 60 )) ]; then
      echo "[$(date -u '+%F %T')] 평가 종료 판정 → pi07-serve 정지"
      systemctl --user stop pi07-serve
      exit 0
    fi
  fi
  sleep 300
done
