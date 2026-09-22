#!/usr/bin/env bash
# π0.7 재현 학습 상태 한눈에: systemd 유닛, 마지막 로그, 검증 결과, 체크포인트, GPU, 메모리.
#   bash src/pi07/status.sh
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

echo "== $(date -u '+%F %T') UTC"
echo "== systemd --user units (pi07*)"
systemctl --user list-units --all --no-pager --plain 'pi07*' 2>/dev/null | grep -E '^pi07' | awk '{printf "  %-34s %-10s %s\n", $1, $3, $4}'

for d in "$RUNS"/train/pi07*; do
  [ -d "$d" ] || continue
  id=$(basename "$d")
  echo "== $id"
  [ -f "$d/COMPLETED" ] && echo "  COMPLETED"
  log="$d/logs/train.log"
  [ -f "$log" ] || continue
  grep -E '^\[.*step [0-9]+/' "$log" | tail -n 1 | sed 's/^/  /'
  grep -E 'VAL step' "$log" | tail -n 3 | sed 's/^/  /'
  n_err=$(grep -c -E 'Traceback|OutOfMemoryError|Killed' "$log")
  [ "$n_err" -gt 0 ] && echo "  ⚠ 로그의 오류 줄 ${n_err}개 (과거 재시작 포함) — 마지막: $(grep -E 'Error' "$log" | tail -n 1 | cut -c1-120)"
  if [ -f "$d/logs/metrics.jsonl" ]; then
    python3 - "$d/logs/metrics.jsonl" <<'PY'
import json, sys, time
rows = [json.loads(l) for l in open(sys.argv[1]) if '"main_anon_gb"' in l]
if len(rows) >= 2:
    a, b = rows[min(5, len(rows) - 1)], rows[-1]   # 워밍업 직후 vs 최신
    ds = b["step"] - a["step"]
    h = sum(r["s_per_step"] for r in rows if a["step"] < r["step"] <= b["step"]) * (ds / max(1, sum(1 for r in rows if a["step"] < r["step"] <= b["step"]))) / 3600
    h = max(h, 1e-6)
    print(f"  mem trend step {a['step']}→{b['step']} (~{h:.1f} h): main {a['main_anon_gb']:.2f}→{b['main_anon_gb']:.2f} G, "
          f"workers {a['workers_pss_gb']:.2f}→{b['workers_pss_gb']:.2f} G "
          f"({(b['workers_pss_gb'] - a['workers_pss_gb']) / h:+.3f} G/h), avail min {min(r['avail_gb'] for r in rows):.1f} G")
PY
  fi
  pid=$(systemctl --user show -p MainPID --value "$id" 2>/dev/null)
  if [ -n "$pid" ] && [ "$pid" != "0" ]; then
    python3 - "$pid" <<'PY'
import sys, glob
main = int(sys.argv[1])
def rollup(p):
    d = {}
    for l in open(f"/proc/{p}/smaps_rollup"):
        k = l.split(":")[0]
        if k in ("Pss_Anon", "Pss_File", "SwapPss", "Rss"):
            d[k] = int(l.split()[1])
    return d
kids = []
for f in glob.glob("/proc/[0-9]*/stat"):
    try:
        x = open(f).read().split()
        if int(x[3]) == main:
            kids.append(int(x[0]))
    except Exception:
        pass
m = rollup(main)
w = [rollup(k) for k in kids]
g = lambda L, k: sum(x.get(k, 0) for x in L) / 1e6
print(f"  live: main anon {m.get('Pss_Anon',0)/1e6:.2f} G swap {m.get('SwapPss',0)/1e6:.2f} G | "
      f"{len(w)} workers anon {g(w,'Pss_Anon'):.2f} G swap {g(w,'SwapPss'):.2f} G file(page cache) {g(w,'Pss_File'):.2f} G")
PY
  fi
  ls -d "$d"/checkpoints/step-* 2>/dev/null | xargs -r -n1 basename | tr '\n' ' ' | sed 's/^/  ckpt: /'; echo
done

echo "== GPU"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | sed 's/^/  /'
echo "== RAM / swap"
free -g | sed 's/^/  /'
