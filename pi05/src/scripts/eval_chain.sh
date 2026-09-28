#!/bin/bash
# 한 차수의 평가를 한 번에 순서대로 돌린다 — 차수를 인자로 받는다 (HANDOFF 40.18).
#
#   setsid nohup systemd-inhibit --what=sleep:idle --why="run10 평가" \
#     bash scripts/eval_chain.sh run10 >> outputs/run10_eval_chain.log 2>&1 < /dev/null &
#   setsid nohup bash scripts/rollout_watchdog.sh >> outputs/run10_eval/watchdog.log 2>&1 < /dev/null &
#
# 9차는 `run9_eval_chain.sh` 로 돌았다. 그 파일은 **고치지 않고 둔다** — 멈춘 4·5단계를
# 같은 명령으로 이어 돌릴 수 있어야 하고, 실험 도중에 실행 코드를 바꾸지 않는다(35.7).
#
# | 단계 | 무엇 | 판 | 시간 |
# |---|---|---:|---:|
# | 0 | 중간 체크포인트 12000·18000 받기 + 세 스텝 구조 검사 | — | 약 30분 |
# | 1 | 언어 검사 (CPU) — 32종 배치로 12000·18000·마지막 | 3회 | 약 1h |
# | 2 | 대조군 — `null_new12` 에 **아직 없는 무대만** (ONLY 로 지정) | 0~2 | 약 30분 |
# | 3 | 신규 12종 — 이번 차수의 목적 | 12 | 약 9h |
# | 4 | 기존 20종 — 망각 검사 | 21 | 약 15h |
# | 5 | 일반화 — run7_unseen 과 같은 13판 | 13 | 약 9h |
#
# ## 언어 검사는 9차 숫자와 바로 견줄 수 있다
#
# 배치 파일(`outputs/probe/batch_pi05_cooking_top32_v5_b16.npz`)과 확산 잡음
# (`jax.random.key(0)`)이 고정이라, 같은 설정이면 **같은 입력**으로 잰다. 그래서
# run9·run7 은 다시 재지 않고 `outputs/run9_eval/probe/` 의 JSON 을 옆에 놓는다.
# ⚠️ 20종 배치(v3batch)는 재지 않는다 — 정규화가 다른 모델끼리는 바닥 오차가 커서
# 방향조차 못 가렸다(40.16 1단계 결과 5번).
#
# ## 2단계 — 대조군은 체크포인트와 상관없어서 한 번만 돌린다
#
# `null_new12` 에는 9차 때 12판이 있다. 이번에 바뀐 무대는 놓기 2종(`*_put_board2`,
# 데모를 ep8652 로 교체 — 40.18)뿐이다. ⚠️ `hotdog_put_micro` 는 9차 대조군이 멈춰서
# actions 파일이 없으므로, ONLY 없이 돌리면 **그 판을 다시 돌다 또 멈출 수 있다.**
# 그래서 ONLY 로 두 판만 준다. 이미 있으면 건너뛴다.
#
# ## 3~5단계의 체크포인트는 3단계가 **시작될 때** 정한다
#
# `outputs/<차수>_eval/ckpt.txt` 에 경로가 있으면 그것을, 없으면 run 폴더의 가장 큰
# 스텝을 쓴다. 1단계 결과를 보고 2단계가 도는 동안 바꿀 수 있게 하려는 것이다.
#
# ## 끊기면
#
# 같은 명령으로 다시 띄운다. 언어 검사는 JSON 이, 롤아웃은 결과 폴더에
# actions_*.npz 가 있으면 건너뛴다(각 스크립트의 원래 동작).
#
# ## 🔴 연쇄 실행 둘을 동시에 돌리지 말 것
#
# 롤아웃 스크립트는 판마다 정책 서버를 `pkill` 로 끄고 새로 띄운다. 둘이 같이 돌면
# **서로의 서버를 죽인다.** 그래서 다른 연쇄 실행이 있으면 시작하지 않고, 다른
# 롤아웃이 돌고 있으면 2단계 전에 끝나기를 기다린다.
#
# ⚠️ 이 파일을 띄우는 명령줄에 정책 서버 프로세스 이름을 literal 로 넣지 말 것 —
# 하위 스크립트의 `pkill -f` 가 이 래퍼까지 죽인다 (종료코드 144).
set -u
cd "$(dirname "$0")/.."
RUNNAME=${1:?"차수 이름을 줄 것 — 예: bash scripts/eval_chain.sh run10"}
CFG=${COOKING_CONFIG:-pi05_cooking_top32_v5}
RUN=checkpoints/$CFG/$RUNNAME
EV=outputs/${RUNNAME}_eval
VENV=third_party/openpi/.venv/bin/python
MIDSTEPS=${MIDSTEPS:-"12000 18000"}
mkdir -p "$EV/probe"
stamp() { date "+%m-%d %H:%M"; }

# ── 다른 연쇄 실행이 있으면 시작하지 않는다 ───────────────────────────────────
# 이 파일끼리는 잠금 파일로 막는다. `pgrep -f` 로 이 파일 이름을 찾으면 `$(...)` 가 만든
# **자기 하위 셸**도 같은 명령줄이라 자기를 "다른 실행" 으로 오인한다.
if pgrep -f "^bash scripts/run9_eval_chain.sh" > /dev/null; then
  echo "❌ run9_eval_chain.sh 가 돌고 있다 — 정책 서버를 서로 죽이므로 시작하지 않는다."
  exit 1
fi
exec 9> outputs/.eval_chain.lock
if ! flock -n 9; then
  echo "❌ 다른 eval_chain.sh 가 돌고 있다 — 정책 서버를 서로 죽이므로 시작하지 않는다."
  exit 1
fi

echo "════════ $RUNNAME 평가 시작 ($CFG) ════════ $(stamp)"
LAST=$(ls -d "$RUN"/[0-9]* 2>/dev/null | grep -v _repair | sort -t/ -k4 -n | tail -1)
if [ -z "$LAST" ]; then
  echo "❌ 체크포인트가 없다: $RUN/<스텝>  — 먼저 받고 finalize_checkpoint.py 로 확인할 것"
  exit 1
fi
echo "마지막 스텝: $LAST"

# ── 0. 중간 체크포인트 받기 + 구조 검사 ───────────────────────────────────────
# 다른 창에서 받고 있을 수 있다. 같은 폴더에 rclone 둘이 동시에 쓰면 꼬이므로 기다린다.
echo; echo "════════ [0/5] 중간 체크포인트 ════════ $(stamp)"
for i in $(seq 1 120); do pgrep -x rclone > /dev/null || break; sleep 30; done
for s in $MIDSTEPS; do
  rclone copy --transfers 8 "gdrive:cooking-vla/$RUN/$s" "$RUN/$s" \
    --exclude "train_state/**" 2>&1 | grep -v NOTICE
done
for d in $(for s in $MIDSTEPS; do echo "$RUN/$s"; done) "$LAST"; do
  s=$(basename "$d")
  if $VENV scripts/verify_checkpoint.py "$d" > "$EV/verify_$s.log" 2>&1; then
    echo "  ✅ $s 구조 검사 통과"
  else
    echo "  ❌ $s 구조 검사 실패 — $EV/verify_$s.log"; tail -5 "$EV/verify_$s.log"
  fi
done

# ── 1. 언어 검사 ──────────────────────────────────────────────────────────────
echo; echo "════════ [1/5] 언어 검사 (CPU, 32종 배치) ════════ $(stamp)"
probe() {   # probe <체크포인트> <이름>
  local ck=$1 tag=$2
  if [ -f "$EV/probe/$tag.json" ]; then echo "  $tag — 이미 있음, 건너뜀"; return; fi
  if [ ! -d "$ck/params" ]; then echo "  ❌ $tag — 체크포인트가 없다: $ck"; return; fi
  echo "  ── $tag  ($ck)  $(stamp)"
  HF_LEROBOT_HOME=$PWD/data/lerobot HF_HUB_OFFLINE=1 \
    $VENV scripts/probe_language.py --config "$CFG" --ckpt "$ck" --batch 16 --device cpu \
      --json-out "$EV/probe/$tag.json" > "$EV/probe/$tag.log" 2>&1
  echo "     종료코드 $?  $(stamp)"
  grep -E "왕복 검사|🔴" "$EV/probe/$tag.log" | sed 's/^/     /'
}
for d in "$LAST" $(for s in $MIDSTEPS; do echo "$RUN/$s"; done); do
  probe "$d" "v5batch_${RUNNAME}_$(basename "$d")"
done
# 앞 차수와 한 표로 — 같은 배치·같은 잡음이라 바로 견줄 수 있다.
$VENV - "$EV/probe" <<'PY'
import glob, json, os, sys
rows = []
for f in sorted(glob.glob("outputs/*_eval/probe/v5batch_*.json")):
    d = json.load(open(f))
    tag = os.path.basename(f)[len("v5batch_"):-len(".json")]
    ie = d["instr_effect"]
    rows.append((tag, d["err"]["true"], ie["all"][1], ie["new12"][1], ie["top20"][1],
                 d["full_effect"][1], d["noise_effect"][1], d["batch_cache"]))
caches = {r[-1] for r in rows}
print(f"\n  {'체크포인트':<16}{'정상 오차':>9}{'지시문만':>9}{'신규12':>9}{'기존20':>9}{'통째':>9}{'잡음':>8}")
for t, e, a, n, o, fu, no, _ in rows:
    print(f"  {t:<16}{e:>9.4f}{a:>+9.1%}{n:>+9.1%}{o:>+9.1%}{fu:>+9.1%}{no:>8.1%}")
if len(caches) > 1:
    print(f"  🔴 배치 파일이 서로 다르다 — 이 표는 견줄 수 없다: {caches}")
print("  (퍼센트 = 정상 오차 대비 늘어난 비율. 신규12 는 7쌍뿐이라 흔들린다 — 40.16)")
PY

# ── 2. 대조군 (없는 무대만) ───────────────────────────────────────────────────
# 다른 롤아웃(예: 따로 띄운 대조군)이 돌고 있으면 끝나기를 기다린다 — 서버를 죽이지 않게.
for i in $(seq 1 240); do
  pgrep -f "^${BEH:-$HOME/miniconda3/envs/behavior/bin/python} scripts/rollout_b1k.py" > /dev/null || break
  [ "$i" = 1 ] && echo "  다른 롤아웃이 돌고 있다 — 끝나기를 기다린다 $(stamp)"
  sleep 30
done
echo; echo "════════ [2/5] 대조군 — 새 무대만 (정책에 안 묻는다) ════════ $(stamp)"
NULL_POLICY=1 ONLY="pepper_put_board2 beet_put_board2" COOKING_CKPT=$LAST COOKING_CONFIG=$CFG \
  OUT=outputs/null_new12 bash scripts/rollout_new12.sh

# ── 3~5. 롤아웃 ───────────────────────────────────────────────────────────────
CK=$(cat "$EV/ckpt.txt" 2>/dev/null || echo "$LAST")
TAG=$(basename "$CK")
echo; echo "════════ 3~5단계 체크포인트: $CK ════════ $(stamp)"
if ! $VENV scripts/verify_checkpoint.py "$CK" > "$EV/verify_rollout_ckpt.log" 2>&1; then
  echo "❌ $CK 가 열리지 않는다 — 3~5단계를 멈춘다. $EV/verify_rollout_ckpt.log"
  exit 1
fi

echo; echo "════════ [3/5] $RUNNAME 신규 12종 ════════ $(stamp)"
COOKING_CKPT=$CK COOKING_CONFIG=$CFG OUT=outputs/${RUNNAME}_${TAG}_new12 bash scripts/rollout_new12.sh

echo; echo "════════ [4/5] $RUNNAME 기존 20종 (망각 검사) ════════ $(stamp)"
COOKING_CKPT=$CK COOKING_CONFIG=$CFG OUT=outputs/${RUNNAME}_${TAG}_base20 bash scripts/rollout_all20.sh

echo; echo "════════ [5/5] $RUNNAME 일반화 (run7_unseen 과 같은 13판) ════════ $(stamp)"
COOKING_CKPT=$CK COOKING_CONFIG=$CFG OUT=outputs/${RUNNAME}_${TAG}_unseen TIER=all \
  bash scripts/rollout_unseen.sh

echo; echo "════════ $RUNNAME 평가 0~5단계 끝 ════════ $(stamp)"
echo "다음: 영상 판독 (판정기보다 영상이 우선 — 40.1). 무대 어긋남은 README 머리가 아니라"
echo "      로그의 [무대 검사] 줄로 셀 것 (9차 결과 폴더는 옛 머리 경고가 섞여 있다 — 40.16)."
