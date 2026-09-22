#!/usr/bin/env python3
"""π0.7 재현 학습기 — HL(고수준 정책) / LL(저수준 VLA) 공용. GPU 한 장.

    python src/pi07/train.py --mode hl --run-id pi07-hl-pi05base-10k --steps 10000
    python src/pi07/train.py --mode ll --run-id pi07-ll-pi05base-30k --steps 30000

세션이 끊겨도 계속 돌도록 systemd 사용자 유닛으로 띄운다 (src/pi07/launch.sh). 이 스크립트 자체는:
- 시작할 때 runs/train/<id>/checkpoints/ 의 최신 체크포인트에서 자동 재개한다 (옵티마이저·스케줄·스텝 포함).
- SIGTERM/SIGINT 를 받으면 현재 스텝을 마치고 체크포인트를 쓴 뒤 0 으로 끝난다.
- runs/train/<id>/STOP 파일이 생겨도 같은 식으로 끝난다.
- 로그마다 메인 프로세스 RssAnon, 워커 Pss 합, MemAvailable, 스왑 사용량을 남긴다.
  (워커 rss 는 fork COW 공유분이 중복 계상되므로 합산하지 않는다 — groot-mt-200k 진단 교훈)
- MemAvailable 과 SwapFree 가 둘 다 바닥이면 커널 OOM 전에 체크포인트를 쓰고 끝낸다 (exit 75).
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import math
import os
import signal
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import B1KIndex, HLDataset, LLDataset, WMDataset, collate  # noqa: E402

CAPSTONE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNS = os.path.join(CAPSTONE, "runs", "train")
EXIT_MEMORY = 75


# ---------------------------------------------------------------- 메모리 계측

def _kb(path: str, keys: tuple[str, ...]) -> dict:
    out = {}
    try:
        for line in open(path):
            k = line.split(":", 1)[0]
            if k in keys:
                out[k] = int(line.split()[1])
    except OSError:
        pass
    return out


def mem_report(worker_pids: list[int]) -> dict:
    me = _kb("/proc/self/status", ("RssAnon", "RssFile", "RssShmem", "VmSwap"))
    # 워커는 익명 메모리(Pss_Anon)와 스왑(SwapPss)만 센다. Pss 전체에는 memmap(lowdim.npy)·비디오 파일의
    # 페이지 캐시가 섞여 있어, 파일을 더 많이 읽을수록 커지지만 회수 가능한 메모리다.
    pss = 0
    for pid in worker_pids:
        r = _kb(f"/proc/{pid}/smaps_rollup", ("Pss_Anon", "SwapPss"))
        pss += r.get("Pss_Anon", 0) + r.get("SwapPss", 0)
    mi = _kb("/proc/meminfo", ("MemAvailable", "SwapTotal", "SwapFree"))
    # RssShmem: CUDA 드라이버의 /dev/zero 공유 매핑 (프로세스당 약 2.3 GB, 일정) — 실제 RAM 이다
    return dict(main_anon_gb=me.get("RssAnon", 0) / 1e6, main_shmem_gb=me.get("RssShmem", 0) / 1e6,
                main_swap_gb=me.get("VmSwap", 0) / 1e6,
                workers_pss_gb=pss / 1e6, n_workers=len(worker_pids),
                avail_gb=mi.get("MemAvailable", 0) / 1e6,
                swap_used_gb=(mi.get("SwapTotal", 0) - mi.get("SwapFree", 0)) / 1e6,
                swap_free_gb=mi.get("SwapFree", 0) / 1e6)


def worker_pids() -> list[int]:
    me = os.getpid()
    out = []
    for d in glob.glob("/proc/[0-9]*/stat"):
        try:
            f = open(d).read().split()
            if int(f[3]) == me:
                out.append(int(f[0]))
        except (OSError, ValueError, IndexError):
            pass
    return out


# ---------------------------------------------------------------- 체크포인트

def ckpt_dirs(run_dir: str) -> list[str]:
    ds = glob.glob(os.path.join(run_dir, "checkpoints", "step-*"))
    ds = [d for d in ds if os.path.exists(os.path.join(d, "DONE"))]
    return sorted(ds, key=lambda d: int(d.rsplit("-", 1)[1]))


def save_ckpt(run_dir: str, step: int, policy, opt, sched, keep_last: int, milestone: int, extra: dict):
    d = os.path.join(run_dir, "checkpoints", f"step-{step:06d}")
    os.makedirs(d, exist_ok=True)
    # CUDA 텐서를 그대로 넘긴다 — torch.save 가 저장소를 하나씩 CPU 로 옮겨 쓰므로 CPU 에 통째 사본이 안 생긴다
    trainable = {n: p.detach() for n, p in policy.named_parameters() if p.requires_grad}
    torch.save(trainable, os.path.join(d, "trainable.pt"))
    torch.save({"opt": opt.state_dict(), "sched": sched.state_dict()}, os.path.join(d, "optim.pt"))
    json.dump(dict(step=step, time=time.strftime("%Y-%m-%d %H:%M:%S"), **extra), open(os.path.join(d, "meta.json"), "w"))
    open(os.path.join(d, "DONE"), "w").close()
    # 오래된 것 정리: 마지막 keep_last 개와 milestone 배수는 남긴다. 옵티마이저 상태는 마지막 것만.
    ds = ckpt_dirs(run_dir)
    for old in ds[:-keep_last]:
        s = int(old.rsplit("-", 1)[1])
        if milestone and s % milestone == 0:
            op = os.path.join(old, "optim.pt")
            if os.path.exists(op):
                os.remove(op)
            continue
        for f in os.listdir(old):
            os.remove(os.path.join(old, f))
        os.rmdir(old)
    return d


def load_ckpt(d: str, policy, opt, sched) -> int:
    sd = torch.load(os.path.join(d, "trainable.pt"), map_location="cpu", weights_only=True)
    params = dict(policy.named_parameters())
    for n, v in sd.items():
        params[n].data.copy_(v.to(params[n].device, params[n].dtype))
    del sd
    op = os.path.join(d, "optim.pt")
    if os.path.exists(op):
        st = torch.load(op, map_location="cpu", weights_only=False)
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        del st
    return int(json.load(open(os.path.join(d, "meta.json")))["step"])


# ---------------------------------------------------------------- 스케줄

def cosine_lambda(warmup: int, total: int, peak: float, final: float):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        p = min(1.0, (step - warmup) / max(1, total - warmup))
        return (final + (peak - final) * 0.5 * (1 + math.cos(math.pi * p))) / peak
    return f


# ---------------------------------------------------------------- 검증

# 액션 23차원의 그룹. 차원 6(torso 4번째)은 q01=q99=0 인 퇴화 차원(정규화값이 항상 -1)이라 그룹 평균에서 뺀다.
LL_GROUPS = (("base", (0, 1, 2)), ("torso", (3, 4, 5)), ("left_arm", tuple(range(7, 14))), ("left_gripper", (14,)),
             ("right_arm", tuple(range(15, 22))), ("right_gripper", (22,)))
LL_HORIZONS = (5, 25)  # 청크 앞 1..h 스텝. 1..50 은 접미사 없는 키(전체)가 그 값이다


def _l1_breakdown(err: torch.Tensor, w: torch.Tensor, key: str) -> dict:
    """절대 오차 err [B, H, D], 패딩 가중 w [B, H, 1] → 전체(1..H, D 차원 전부) · 지평별 · 그룹별 L1."""
    D = err.shape[-1]
    out = {key: float((err * w).sum() / (w.sum() * D))}
    for h in LL_HORIZONS:
        out[f"{key}_h{h}"] = float((err[:, :h] * w[:, :h]).sum() / (w[:, :h].sum() * D))
    for gname, dims in LL_GROUPS:
        out[f"{key}_{gname}"] = float((err[..., dims] * w).sum() / (w.sum() * len(dims)))
    return out


@torch.no_grad()
def _val_ll_batch(policy, name: str, b: dict, g: torch.Generator, g_sample: torch.Generator) -> tuple[dict, torch.Tensor]:
    """배치 하나의 지표(배치 안에서는 패딩 마스크 가중)와 표본별 L1 [B]."""
    pad, kv = policy.encode_prefix(b["images"], b["prompt"], b["state"].numpy(), b.get("goal_images"))
    a = b["actions"].to(policy.device)
    B, H, D = a.shape
    actions = torch.nn.functional.pad(a, (0, 32 - D))
    noise = torch.randn(actions.shape, generator=g, device=policy.device)
    t = torch.rand(B, generator=g, device=policy.device) * 0.998 + 0.001
    x_t = t[:, None, None] * noise + (1 - t[:, None, None]) * actions
    with torch.autocast("cuda", dtype=torch.bfloat16):
        v = policy.expert_velocity(pad, kv, x_t, t)
    err = (v.float() - (noise - actions))[..., :D].pow(2)
    w = (~b["action_is_pad"].to(policy.device)).float()[..., None]
    m = {f"val_{name}": float((err * w).sum() / (w.sum() * D))}
    # 5스텝 적분 결과와 정답 청크(50 스텝)의 L1 (정규화 공간). 초기 노이즈는 g_sample 로 고정한다
    pred = policy.sample_actions(b["images"], b["prompt"], b["state"].numpy(), b.get("goal_images"),
                                 generator=g_sample)
    l1 = (pred - a).abs()
    m.update(_l1_breakdown(l1, w, f"val_{name}_l1"))
    if "prev_action" in b:  # 기준선: 직전 액션 a[t-1] 을 50 스텝 반복 (fixed_val_ll 이 싣는다)
        rep = (b["prev_action"].to(policy.device)[:, None] - a).abs()
        m.update(_l1_breakdown(rep, w, f"val_{name}_rep_l1"))
    return m, ((l1 * w).sum((1, 2)) / (w.sum((1, 2)) * D)).cpu()


@torch.no_grad()
def val_ll(policy, batches: list[dict]) -> dict:
    """고정 표본·고정 잡음·고정 t 로 flow matching 손실과 5스텝 적분 L1 — 스텝 간 비교가 가능하도록.

    스트림(nogoal / goal)별로 배치 값을 표본 수로 가중 평균한다. 키: val_<스트림> (flow 손실),
    val_<스트림>_l1[_h5|_h25|_<그룹>] (모델), val_<스트림>_rep_l1[…] (직전 액션 반복 기준선), val_<스트림>_n.
    잡음 생성기는 스트림마다 같은 시드에서 다시 시작한다 — fixed_val_ll 이 goal 배치 k 를 nogoal 배치 k 와 같은
    표본으로 만들므로 둘은 같은 잡음·같은 t 를 받는다. val_goal_minus_nogoal_l1 은 그 짝의 표본별 L1 차이의 평균(±se)."""
    acc: dict[str, dict] = {}
    n: dict[str, int] = {}
    gens: dict[str, tuple] = {}
    per: dict[str, list] = {}
    ids: dict[str, list] = {}
    for name, b in batches:
        if name not in gens:
            # flow 손실용 생성기(1234)는 이전 코드와 소비 순서가 같다. 샘플링 노이즈는 별도 생성기에서 뽑는다
            gens[name] = (torch.Generator(device=policy.device).manual_seed(1234),
                          torch.Generator(device=policy.device).manual_seed(4321))
        m, l1 = _val_ll_batch(policy, name, b, *gens[name])
        B = len(l1)
        sums = acc.setdefault(name, {})
        for k, x in m.items():
            sums[k] = sums.get(k, 0.0) + x * B
        n[name] = n.get(name, 0) + B
        per.setdefault(name, []).append(l1)
        ids.setdefault(name, []).extend(zip(b.get("episode", [None] * B), b.get("t", [None] * B)))
    out = {}
    for name in acc:
        out.update({k: x / n[name] for k, x in acc[name].items()})
        out[f"val_{name}_n"] = n[name]
    # 서브골 유/무 짝 차이: goal 표본이 nogoal 앞부분과 같은 (에피소드, t) 일 때만 낸다
    pg, pn = ids.get("goal", []), ids.get("nogoal", [])
    if pg and None not in pg[0] and pg == pn[:len(pg)]:
        d = torch.cat(per["goal"]) - torch.cat(per["nogoal"])[:len(pg)]
        out["val_goal_minus_nogoal_l1"] = float(d.mean())
        if len(d) > 1:
            out["val_goal_minus_nogoal_l1_se"] = float(d.std() / math.sqrt(len(d)))
    return out


@torch.no_grad()
def val_hl(policy, batches: list[dict]) -> dict:
    """탐욕 생성 정확도 — 전체 / 전환 표본(이력의 마지막 ≠ 정답) / 유지 표본.
    유지 기준선 = 이력의 마지막을 그대로 내는 것 (전환 표본에서 전부 틀린다)."""
    was = policy.training
    policy.eval()
    n = ok = ns = oks = 0
    loss_sum = acc_sum = 0.0
    for b in batches:
        l, info = policy.loss(b)
        loss_sum += float(l)
        acc_sum += float(info["tok_acc"])
        gen = policy.generate(b["images"], b["prompt"])
        for g_, tgt, sw in zip(gen, b["target"], b["switch"]):
            n += 1
            ok += int(g_ == tgt)
            if sw:
                ns += 1
                oks += int(g_ == tgt)
    policy.train(was)
    return dict(val_loss=loss_sum / len(batches), val_tok_acc=acc_sum / len(batches), val_exact=ok / max(n, 1),
                val_exact_switch=oks / max(ns, 1), val_exact_stay=(ok - oks) / max(n - ns, 1),
                val_stay_baseline=(n - ns) / max(n, 1), val_n=n, val_n_switch=ns)


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = ((a.float() - b.float()) / 255.0).pow(2).mean().item()
    return float(10 * math.log10(1.0 / max(mse, 1e-10)))


@torch.no_grad()
def val_wm(policy, batches: list[dict], out_png: str) -> dict:
    from world_model import tile, wm_prompt
    """고정 σ·고정 잡음 손실 + 생성 PSNR. 기준선 = 현재 관측을 그대로 서브골로 쓰는 것(복사)."""
    was = policy.training
    policy.eval()
    g = torch.Generator(device=policy.device).manual_seed(1234)
    losses = []
    for b in batches:
        ref, hw = policy.encode(tile(b["images"]))
        x0, _ = policy.encode(tile(b["goal_images"]))
        txt = policy.text([wm_prompt(i, s_) for i, s_ in zip(b["instruction"], b["subtask"])])
        for sv in (0.25, 0.5, 0.75):
            noise = torch.randn(x0.shape, generator=g, device=policy.device)
            sig = torch.full((x0.shape[0],), sv, device=policy.device)
            x_t = (1 - sv) * x0 + sv * noise
            losses.append(float(torch.nn.functional.mse_loss(policy.velocity(x_t, ref, sig, txt, hw), noise - x0)))
    b = batches[0]
    pred = policy.generate(b["images"], b["instruction"], b["subtask"]).cpu()
    ps_pred = _psnr(pred, b["goal_images"])
    ps_copy = _psnr(b["images"], b["goal_images"])
    try:
        from PIL import Image
        rows = []
        for i in range(pred.shape[0]):
            r = [torch.cat(list(x[i]), dim=1) for x in (b["images"], pred, b["goal_images"])]
            rows.append(torch.cat(r, dim=1))
        Image.fromarray(torch.cat(rows, 0).numpy()).save(out_png)
    except Exception as ex:  # 그림 저장 실패는 학습을 막지 않는다
        print("png save failed:", ex)
    policy.train(was)
    return dict(val_loss=float(np.mean(losses)), val_psnr_pred=ps_pred, val_psnr_copy=ps_copy)


def fixed_val_wm(idx, n: int, bs: int) -> list:
    ds = WMDataset("val", index=idx, seed=999)
    ds._setup()
    samples = [ds.sample() for _ in range(n)]
    ds.reader.close()
    return [collate(samples[i:i + bs]) for i in range(0, n, bs)]


def fixed_val_ll(idx, n_nogoal: int, n_goal: int, bs: int) -> list:
    """nogoal n_nogoal 개 + goal n_goal 개. goal 표본은 nogoal 앞 n_goal 개와 같은 (에피소드, t) · 같은 프롬프트에
    서브골 3뷰만 붙인 것이고 배치 경계도 같다 — val_ll 이 서브골 유/무를 같은 표본·같은 잡음으로 짝비교한다.
    표본마다 "직전 액션 a[t-1] 반복" 기준선용 prev_action (정규화, clip ±1.5) 을 싣는다."""
    assert n_goal <= n_nogoal, (n_goal, n_nogoal)
    kw = dict(index=idx, seed=999, p_meta_drop=0.0, p_meta_item_drop=0.0,
              p_subtask_drop_goal=0.0, p_subtask_drop_nogoal=0.0)
    ds = LLDataset("val", with_goal=False, **kw)
    ds._setup()
    nogoal = [ds.sample() for _ in range(n_nogoal)]
    ds.reader.close()
    by_ep = {e["episode_index"]: e for e in ds.eps}
    ds = LLDataset("val", with_goal=True, **kw)
    ds._setup()
    goal = [ds.make(by_ep[s["episode"]], s["t"]) for s in nogoal[:n_goal]]
    ds.reader.close()
    for s in nogoal + goal:
        e = by_ep[s["episode"]]
        # t=0 에는 직전 액션이 없어 a[0] 을 쓴다 (프레임 균등 표집이라 에피소드 길이분의 1 확률)
        prev = np.asarray(idx.lowdim[e["offset"] + max(s["t"] - 1, 0), :idx.action_dim], np.float32)
        s["prev_action"] = idx.norm(prev, "action").astype(np.float32)
    out = []
    for name, samples in (("nogoal", nogoal), ("goal", goal)):
        for i in range(0, len(samples), bs):
            out.append((name, collate(samples[i:i + bs])))
    return out


def fixed_val_hl(idx, n: int, bs: int, n_edge: int = 64) -> list:
    """검증 에피소드 표본, 질의 간격 Δ=1초 (런타임과 같음).
    - n 개: 1초 격자 균등 (실제 런타임 분포 — 대부분 '유지')
    - n_edge 개: 경계 직후 0~1초 (전환이어야 맞음), n_edge 개: 경계 직전 0~1초 (유지해야 맞음, 어려운 음성)
    switch = 이력의 마지막 지시와 정답이 다른 표본."""
    ds = HLDataset("val", index=idx, seed=999, history_dropout=0.0, boundary_frac=0.0)
    ds._setup()
    rng = np.random.default_rng(999)
    samples = []
    eps = ds.eps
    for _ in range(n):
        e = eps[int(rng.integers(len(eps)))]
        t = int(rng.integers(e["length"] // idx.fps)) * idx.fps
        samples.append(ds.make(e, t, idx.fps))
    for sign in (+1, -1):
        for _ in range(n_edge):
            e = eps[int(rng.integers(len(eps)))]
            k = int(rng.integers(1, len(e["segments"])))
            b = e["segments"][k]["start"]
            t = b + sign * int(rng.uniform(0.0, 1.0) * idx.fps) - (1 if sign < 0 else 0)
            t = min(max(t, 0), e["length"] - 1)
            samples.append(ds.make(e, t, idx.fps))
    ds.reader.close()
    return [collate(samples[i:i + bs]) for i in range(0, len(samples), bs)]


# ---------------------------------------------------------------- 메인

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hl", "ll", "wm"], required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2, help="스텝당 마이크로배치 수 (유효 배치 = micro × accum)")
    ap.add_argument("--goal-per-micro", type=int, default=4, help="LL: 마이크로배치 중 서브골 예시 수 (π0.7: 25%)")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--final-lr", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--keep-last", type=int, default=2)
    ap.add_argument("--milestone", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-from", default=None, help="자기 체크포인트가 없을 때 이 체크포인트의 학습 파라미터로 시작 (옵티마이저는 새로)")
    ap.add_argument("--gen-goal-dir", default=None, help="LL: 생성 서브골 디렉토리 (gen_subgoals.py). 서브골 예시의 --p-gen 만큼 사용")
    ap.add_argument("--p-gen", type=float, default=0.5)
    ap.add_argument("--duty", type=float, default=1.0,
                    help="GPU 가동률 상한 (0<d≤1). 스텝마다 연산 시간 × (1/d − 1) 만큼 쉰다 — sudo 없이 평균 전력·온도를 낮추는 용도")
    ap.add_argument("--min-avail-gb", type=float, default=2.0)
    ap.add_argument("--min-swap-free-gb", type=float, default=8.0)
    args = ap.parse_args()

    run_dir = os.path.join(RUNS, args.run_id)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    metrics_f = open(os.path.join(run_dir, "logs", "metrics.jsonl"), "a")

    def log(msg: str):
        tag = "" if int(os.environ.get("RANK", "0")) == 0 else f"[r{os.environ['RANK']}] "
        print(f"[{time.strftime('%m-%d %H:%M:%S')}] {tag}{msg}", flush=True)

    stop = {"flag": False, "why": ""}

    def on_signal(sig, frm):
        stop["flag"] = True
        stop["why"] = signal.Signals(sig).name

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # 데이터 병렬 (torchrun --nproc_per_node=N). N=1 이면 아래 분기는 전부 꺼진다.
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        import torch.distributed as dist
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
    device = torch.device("cuda", local_rank if world > 1 else 0)
    is_main = rank == 0
    # 모델 쪽 RNG(흐름 매칭의 잡음·시간)는 rank 마다 다르게, 데이터 시드도 rank 로 갈린다
    torch.manual_seed(args.seed + 1000 * rank)

    idx = B1KIndex()
    if args.mode == "hl":
        from models import HLPolicy
        policy = HLPolicy(device)
        lr = args.lr or 1e-4
        final_lr = args.final_lr or 1e-5
    elif args.mode == "ll":
        from models import LLPolicy
        policy = LLPolicy(device)
        lr = args.lr or 2.5e-5  # openpi π0.5 파인튜닝 기본값
        final_lr = args.final_lr or 2.5e-6
    else:
        from world_model import WorldModel
        policy = WorldModel(device)
        lr = args.lr or 1e-4
        final_lr = args.final_lr or 1e-5
    policy.train()
    if args.mode == "ll":
        policy.model.paligemma_with_expert.paligemma.eval()
    params = [p for p in policy.parameters() if p.requires_grad]
    log(f"mode={args.mode} trainable={sum(p.numel() for p in params) / 1e6:.1f}M "
        f"gpu_alloc={torch.cuda.memory_allocated() / 1e9:.1f}GB")
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_lambda(args.warmup, args.steps, lr, final_lr))

    step = 0
    ds = ckpt_dirs(run_dir)
    if ds:
        step = load_ckpt(ds[-1], policy, opt, sched)
        log(f"resumed from {ds[-1]} (step {step})")
    elif args.init_from:
        sd = torch.load(os.path.join(args.init_from, "trainable.pt"), map_location="cpu", weights_only=True)
        pmap = dict(policy.named_parameters())
        for n, v in sd.items():
            pmap[n].data.copy_(v.to(pmap[n].device, pmap[n].dtype))
        del sd
        log(f"initialized trainable params from {args.init_from}")
    if world > 1:  # 적재 경로가 같아도 한 번 맞춰 둔다 — 이후 그래디언트 평균으로 계속 같이 간다
        for p_ in params:
            dist.broadcast(p_.data, 0)
    if step >= args.steps:
        log("이미 완료됨")
        return

    # 검증 표본은 학습 워커를 띄우기 전에 고정해 둔다
    if args.mode == "ll":
        val_batches = fixed_val_ll(idx, 96, 32, 16)
    elif args.mode == "hl":
        val_batches = fixed_val_hl(idx, 256, 16)
    else:
        val_batches = fixed_val_wm(idx, 32, args.micro_batch)
        os.makedirs(os.path.join(run_dir, "val"), exist_ok=True)
    log(f"val batches fixed: {len(val_batches)}")

    common = dict(num_workers=args.workers, collate_fn=collate, persistent_workers=True,
                  prefetch_factor=2, pin_memory=False, multiprocessing_context="fork")
    if args.mode == "hl":
        loaders = [(1.0, iter(DataLoader(HLDataset("train", seed=args.seed, rank=rank, start_step=step, index=idx),
                                          batch_size=args.micro_batch, **common)))]
    elif args.mode == "wm":
        loaders = [(1.0, iter(DataLoader(WMDataset("train", seed=args.seed, rank=rank, start_step=step, index=idx),
                                          batch_size=args.micro_batch, **common)))]
    else:
        n_goal = args.goal_per_micro
        n_plain = args.micro_batch - n_goal
        w_goal = max(1, round(args.workers * n_goal / args.micro_batch))
        loaders = [
            (n_plain / args.micro_batch,
             iter(DataLoader(LLDataset("train", with_goal=False, seed=args.seed, rank=rank, start_step=step, index=idx),
                             batch_size=n_plain, **{**common, "num_workers": args.workers - w_goal}))),
            (n_goal / args.micro_batch,
             iter(DataLoader(LLDataset("train", with_goal=True, seed=args.seed + 1, rank=rank, start_step=step, index=idx,
                                       gen_dir=args.gen_goal_dir, p_gen=args.p_gen),
                             batch_size=n_goal, **{**common, "num_workers": w_goal}))),
        ]

    t_last = time.time()
    wait_acc = 0.0
    loss_acc = []
    tok_acc = []
    pids = []
    while step < args.steps:
        t_step0 = time.time()
        for _ in range(args.accum):
            for w, it in loaders:
                t0 = time.time()
                b = next(it)
                wait_acc += time.time() - t0
                loss, info = policy.loss(b)
                (loss * w / args.accum).backward()
                loss_acc.append(float(info["loss"]))
                if "tok_acc" in info:
                    tok_acc.append(float(info["tok_acc"]))
        if world > 1:
            # 평탄화해서 한 번에 all-reduce (430M × 4B = 1.7 GB). grad 가 없는 파라미터는 0 으로 채워
            # 두 rank 의 텐서 목록·순서를 같게 만든다
            for p_ in params:
                if p_.grad is None:
                    p_.grad = torch.zeros_like(p_)
            gs = [p_.grad for p_ in params]
            flat = torch._utils._flatten_dense_tensors(gs)
            dist.all_reduce(flat)
            flat.div_(world)
            for g_, src in zip(gs, torch._utils._unflatten_dense_tensors(flat, gs)):
                g_.copy_(src)
            del flat
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if args.duty < 1.0:
            torch.cuda.synchronize()
            time.sleep((time.time() - t_step0) * (1.0 / max(args.duty, 0.1) - 1.0))

        if step % args.log_every == 0 or step == 1:
            if not pids or step % (args.log_every * 10) == 0:
                pids = worker_pids()
            dt = (time.time() - t_last) / (args.log_every if step > 1 else 1)
            m = mem_report(pids)
            rec = dict(step=step, loss=float(np.mean(loss_acc)), lr=sched.get_last_lr()[0], grad_norm=float(gn),
                       s_per_step=dt, data_wait_s=wait_acc / (args.log_every if step > 1 else 1),
                       gpu_peak_gb=torch.cuda.max_memory_allocated() / 1e9, **m)
            if tok_acc:
                rec["tok_acc"] = float(np.mean(tok_acc))
            if is_main:
                metrics_f.write(json.dumps(rec) + "\n")
                metrics_f.flush()
            eta_h = (args.steps - step) * dt / 3600
            log(f"step {step}/{args.steps} loss {rec['loss']:.4f}"
                + (f" tok_acc {rec['tok_acc']:.3f}" if tok_acc else "")
                + f" lr {rec['lr']:.2e} gn {rec['grad_norm']:.2f} {dt:.2f}s/it wait {rec['data_wait_s']:.2f}s "
                f"| mem main {m['main_anon_gb']:.1f}G+shm {m['main_shmem_gb']:.1f}G workers {m['workers_pss_gb']:.1f}G "
                f"avail {m['avail_gb']:.1f}G swap {m['swap_used_gb']:.1f}G | eta {eta_h:.1f}h")
            loss_acc, tok_acc = [], []
            wait_acc = 0.0
            t_last = time.time()
            if m["avail_gb"] < args.min_avail_gb and m["swap_free_gb"] < args.min_swap_free_gb:
                stop["flag"] = True
                stop["why"] = f"memory avail={m['avail_gb']:.1f}G swap_free={m['swap_free_gb']:.1f}G"

        if os.path.exists(os.path.join(run_dir, "STOP")):
            stop["flag"] = True
            stop["why"] = "STOP file"
        if world > 1:  # 어느 rank 든 멈추기로 했으면 다 같이 멈춘다 — 한쪽만 나가면 다른 쪽이 all-reduce 에서 멈춘다
            f = torch.tensor([1.0 if stop["flag"] else 0.0], device=device)
            dist.all_reduce(f, op=dist.ReduceOp.MAX)
            if f.item() > 0 and not stop["flag"]:
                stop["flag"] = True
                stop["why"] = "다른 rank 의 중단 요청"

        if (step % args.val_every == 0 or step == args.steps) and is_main:
            tv = time.time()
            # 검증이 죽어도 학습은 계속한다 — 검증은 저장보다 먼저 돌아서, 여기서 예외가 나면 같은 스텝에서
            # 재시작 → 같은 예외가 반복된다
            try:
                if args.mode == "ll":
                    v = val_ll(policy, val_batches)
                elif args.mode == "hl":
                    v = val_hl(policy, val_batches)
                else:
                    v = val_wm(policy, val_batches, os.path.join(run_dir, "val", f"step-{step:06d}.png"))
            except Exception as ex:  # noqa: BLE001
                import traceback
                log(f"VAL 실패 (학습은 계속): {type(ex).__name__}: {ex}\n{traceback.format_exc()}")
                torch.cuda.empty_cache()
                v = {"val_error": str(ex)[:200]}
            policy.train()
            if args.mode == "ll":
                policy.model.paligemma_with_expert.paligemma.eval()
            metrics_f.write(json.dumps(dict(step=step, **v)) + "\n")
            metrics_f.flush()
            log(f"VAL step {step} " + " ".join(f"{k}={x:.4f}" if isinstance(x, float) else f"{k}={x}"
                                             for k, x in v.items()) + f" ({time.time() - tv:.0f}s)")
            t_last += time.time() - tv

        if (step % args.save_every == 0 or step == args.steps or stop["flag"]) and is_main:
            ts = time.time()
            d = save_ckpt(run_dir, step, policy, opt, sched, args.keep_last, args.milestone,
                          dict(mode=args.mode, run_id=args.run_id, total_steps=args.steps))
            log(f"saved {d} ({time.time() - ts:.0f}s)")
            t_last += time.time() - ts

        if world > 1 and (step % args.val_every == 0 or step % args.save_every == 0 or step == args.steps):
            dist.barrier()   # rank 0 의 검증·저장을 기다린다

        if stop["flag"]:
            log(f"중단: {stop['why']} — step {step} 에서 체크포인트 저장 후 종료")
            code = EXIT_MEMORY if stop["why"].startswith("memory") else 0
            os._exit(code)  # 워커 정리 대기 없이 끝낸다 (체크포인트는 이미 DONE)

    log("학습 완료")
    if not is_main:
        os._exit(0)
    open(os.path.join(run_dir, "COMPLETED"), "w").close()
    ry = os.path.join(run_dir, "run.yaml")
    if os.path.exists(ry):  # status: running → done (다른 줄은 건드리지 않는다)
        lines = open(ry).read().split("\n")
        lines = [("status: done" + ln[len("status: running"):]) if ln.startswith("status: running") else ln
                 for ln in lines]
        open(ry, "w").write("\n".join(lines))
    os._exit(0)


if __name__ == "__main__":
    main()
