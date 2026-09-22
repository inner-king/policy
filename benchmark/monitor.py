#!/usr/bin/env python
"""Run a command and sample its resource usage (process tree + system + GPU).

usage: monitor.py --out DIR --name NAME [--interval 0.5] -- cmd args...

Child stdout lines of the form "@@MARK <name>" record phase timestamps;
"@@STAT <json>" lines are stored verbatim in the summary.
"""
import argparse, json, os, subprocess, sys, threading, time
import psutil


def gpu_query():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        mem, util, pwr, temp = [x.strip() for x in out.strip().split(",")]
        return float(mem), float(util), float(pwr), float(temp)
    except Exception:
        return None


def gpu_procs():
    """Per-process GPU memory (compute + graphics) from the plain nvidia-smi table."""
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return {}
    res, in_tbl = {}, False
    for line in out.splitlines():
        if "Processes:" in line:
            in_tbl = True
            continue
        if in_tbl and "MiB" in line:
            parts = line.strip("| ").split()
            try:
                pid = int(parts[3]); typ = parts[4]; mib = float(parts[-1].replace("MiB", ""))
                res[pid] = (typ, mib)
            except Exception:
                pass
    return res


def pss_of(p):
    try:
        with open(f"/proc/{p.pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Pss:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        return 0
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    os.makedirs(a.out, exist_ok=True)

    # ---- baseline (3 s) ----
    base_sys, base_gpu, base_swap = [], [], []
    for _ in range(6):
        vm = psutil.virtual_memory(); base_sys.append(vm.total - vm.available)
        base_swap.append(psutil.swap_memory().used)
        g = gpu_query();  base_gpu.append(g[0] if g else 0)
        time.sleep(0.5)
    b_sys, b_gpu, b_swap = min(base_sys), min(base_gpu), min(base_swap)

    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1, start_new_session=True, errors="replace")
    root = psutil.Process(proc.pid)
    marks, stats = {}, []
    logf = open(os.path.join(a.out, f"{a.name}.log"), "w")

    def reader():
        for line in proc.stdout:
            t = time.time() - t0
            logf.write(f"[{t:8.2f}] {line}"); logf.flush()
            s = line.strip()
            if "@@MARK" in s:
                marks[s.split("@@MARK", 1)[1].strip()] = t
            elif "@@STAT" in s:
                try: stats.append(json.loads(s.split("@@STAT", 1)[1]))
                except Exception: stats.append(s)
    th = threading.Thread(target=reader, daemon=True); th.start()

    gpu_latest = {"v": None, "procs": {}}
    stop = threading.Event()
    def gpu_loop():
        n = 0
        while not stop.is_set():
            gpu_latest["v"] = gpu_query()
            if n % 4 == 0:
                gpu_latest["procs"] = gpu_procs()
            n += 1
            time.sleep(max(a.interval, 0.5))
    gt = threading.Thread(target=gpu_loop, daemon=True); gt.start()

    rows, last_cpu, last_t, n = [], None, None, 0
    child_names = {}
    peak_proc_gpu = 0.0
    while proc.poll() is None:
        now = time.time() - t0
        if now > a.timeout:
            os.killpg(proc.pid, 9); break
        try:
            procs = [root] + root.children(recursive=True)
        except psutil.NoSuchProcess:
            break
        rss = cpu = rd = wr = thr = 0
        try: root_rss = root.memory_info().rss
        except psutil.NoSuchProcess: root_rss = 0
        for p in procs[1:]:
            if p.pid not in child_names:
                try: child_names[p.pid] = " ".join(p.cmdline())[:160]
                except Exception: child_names[p.pid] = "?"
        for p in procs:
            try:
                with p.oneshot():
                    rss += p.memory_info().rss
                    ct = p.cpu_times(); cpu += ct.user + ct.system + ct.children_user + ct.children_system
                    thr += p.num_threads()
                    try:
                        io = p.io_counters(); rd += io.read_bytes; wr += io.write_bytes
                    except Exception: pass
            except psutil.NoSuchProcess:
                pass
        pss = sum(pss_of(p) for p in procs) if n % 4 == 0 else (rows[-1]["pss_mb"] * 2**20 if rows else 0)
        cores = (cpu - last_cpu) / (now - last_t) if last_cpu is not None and now > last_t else 0.0
        last_cpu, last_t = cpu, now
        vm = psutil.virtual_memory(); sw = psutil.swap_memory()
        sys_cpu = psutil.cpu_percent(interval=None)
        g = gpu_latest["v"]
        pids = {p.pid for p in procs}
        pg = sum(m for pid, (typ, m) in gpu_latest["procs"].items() if pid in pids)
        peak_proc_gpu = max(peak_proc_gpu, pg)
        rows.append(dict(
            t=round(now, 2), rss_mb=rss / 2**20, root_rss_mb=root_rss / 2**20, pss_mb=pss / 2**20, cpu_cores=cores, threads=thr, nprocs=len(procs),
            read_mb=rd / 2**20, write_mb=wr / 2**20,
            sys_used_delta_mb=(vm.total - vm.available - b_sys) / 2**20,
            swap_delta_mb=(sw.used - b_swap) / 2**20, sys_cpu_pct=sys_cpu,
            gpu_mem_delta_mb=(g[0] - b_gpu) if g else None, gpu_util=g[1] if g else None,
            gpu_power_w=g[2] if g else None, gpu_temp=g[3] if g else None, gpu_proc_mb=pg,
        ))
        n += 1
        time.sleep(a.interval)
    rc = proc.wait()
    stop.set(); th.join(timeout=10); logf.close()
    wall = time.time() - t0

    import csv
    with open(os.path.join(a.out, f"{a.name}.csv"), "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    def window(t_a, t_b):
        return [r for r in rows if t_a <= r["t"] <= t_b]

    def agg(rs):
        if not rs: return None
        def mx(k): v = [r[k] for r in rs if r[k] is not None]; return round(max(v), 1) if v else None
        def av(k): v = [r[k] for r in rs if r[k] is not None]; return round(sum(v) / len(v), 2) if v else None
        return dict(duration_s=round(rs[-1]["t"] - rs[0]["t"], 1),
                    rss_mb_max=mx("rss_mb"), root_rss_mb_max=mx("root_rss_mb"), pss_mb_max=mx("pss_mb"), sys_used_delta_mb_max=mx("sys_used_delta_mb"),
                    swap_delta_mb_max=mx("swap_delta_mb"), gpu_mem_delta_mb_max=mx("gpu_mem_delta_mb"),
                    gpu_proc_mb_max=mx("gpu_proc_mb"),
                    cpu_cores_avg=av("cpu_cores"), cpu_cores_max=mx("cpu_cores"), gpu_util_avg=av("gpu_util"),
                    gpu_util_max=mx("gpu_util"), gpu_power_w_avg=av("gpu_power_w"), gpu_power_w_max=mx("gpu_power_w"),
                    threads_max=mx("threads"), nprocs_max=mx("nprocs"))

    phases = {}
    ms = sorted(marks.items(), key=lambda kv: kv[1])
    edges = [("start", 0.0)] + ms + [("exit", wall)]
    for (na, ta), (nb, tb) in zip(edges, edges[1:]):
        phases[f"{na}->{nb}"] = agg(window(ta, tb))

    summary = dict(name=a.name, cmd=cmd, returncode=rc, wall_s=round(wall, 1), marks={k: round(v, 1) for k, v in marks.items()},
                   stats=stats, baseline=dict(sys_used_mb=b_sys / 2**20, gpu_mem_mb=b_gpu, swap_mb=b_swap / 2**20),
                   overall=agg(rows), phases=phases,
                   disk_read_mb_total=round(max(r["read_mb"] for r in rows), 1) if rows else None,
                   disk_write_mb_total=round(max(r["write_mb"] for r in rows), 1) if rows else None,
                   peak_gpu_proc_mb=peak_proc_gpu, child_cmds=sorted({c.replace(os.path.expanduser("~"), "~") for c in child_names.values()}))
    with open(os.path.join(a.out, f"{a.name}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    sys.exit(rc)


if __name__ == "__main__":
    main()
