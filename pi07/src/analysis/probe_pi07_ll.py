#!/usr/bin/env python3
"""π0.7 저수준 정책(LL) 진단 — 체크포인트를 검증 분할에 직접 돌려 무엇을 배웠는지 나눠 본다.

    vla/.venv/bin/python src/analysis/probe_pi07_ll.py --ckpt runs/train/pi07ll-pi05base-30k/checkpoints/step-030000

측정 (전부 검증 에피소드 180개에서, 학습에 쓰지 않은 프레임):
  1) 관절군별 L1 — 모델 vs 두 기준선(직전 액션 유지 / 정규화 중앙값)
  2) 청크 안 스텝별 L1 — 50스텝 동안 오차가 어떻게 자라는가
  3) 조건 민감도 — 같은 잡음으로 프롬프트만 바꿔 예측이 얼마나 달라지는가
     (같은 프롬프트 · 다른 잡음 = 바닥값. 이보다 작으면 그 조건을 무시하는 것이다)
산출: <run>/analysis/ll_probe.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

CAPSTONE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(CAPSTONE, "src", "pi07"))

from dataset import B1KIndex, LLDataset, collate, pi07_prompt  # noqa: E402

GROUPS = [("베이스", 0, 3), ("토르소", 3, 7), ("왼팔", 7, 14), ("왼그리퍼", 14, 15),
          ("오른팔", 15, 22), ("오른그리퍼", 22, 23)]
H = 50
D = 23


def step_delta(a: torch.Tensor) -> torch.Tensor:
    """청크 안 스텝당 |증분| — 떨림의 진폭 (부호 전환 비율은 진폭과 무관하므로 둘 다 본다)."""
    return torch.diff(a, dim=1).abs().mean(1)


def flip_rate(a: torch.Tensor) -> torch.Tensor:
    """청크 [B,H,D] 안에서 증분의 부호가 바뀌는 비율 — 롤아웃 액션 로그에 쓴 것과 같은 떨림 척도."""
    d = torch.diff(a, dim=1)
    return (torch.diff(torch.sign(d), dim=1) != 0).float().mean(1)


def sample_with_noise(policy, b, noise, prompts=None, goal=None, steps=5):
    """주어진 잡음에서 5스텝 오일러 적분. 잡음을 고정해야 프롬프트만의 효과를 볼 수 있다."""
    pad, kv = policy.encode_prefix(b["images"], prompts or b["prompt"], b["state"].numpy(), goal)
    x = noise.clone()
    dt = -1.0 / steps
    for i in range(steps):
        t = torch.full((x.shape[0],), 1.0 + i * dt, device=policy.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = policy.expert_velocity(pad, copy.deepcopy(kv), x, t)
        x = x + dt * v.float()
    return x[..., :D]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/train/pi07ll-pi05base-30k/checkpoints/step-030000")
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--k", type=int, default=8, help="표본 K개의 중앙값/평균/최선값도 잰다")
    ap.add_argument("--steps-sweep", default="5,10,20", help="디노이즈 스텝 수 비교 (런타임은 5)")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda")
    idx = B1KIndex()

    # 검증 표본: 드롭아웃 없음(항상 subtask+메타 전부), 서브골 실제 미래 프레임
    ds = LLDataset("val", with_goal=True, index=idx, seed=999, p_meta_drop=0.0, p_meta_item_drop=0.0,
                   p_subtask_drop_goal=0.0)
    ds._setup()
    samples = []
    while len(samples) < args.n:
        e = ds._pick_episode()
        t = ds.rng.randrange(e["length"])
        if t < 1:
            continue
        si = idx.segment_at(e, t)
        seg = e["segments"][si]
        rec = dict(images=ds._images(e, t), state=idx.norm(idx.state(e, t), "state").astype(np.float32))
        a, pad = idx.action_chunk(e, t, H)
        rec["actions"] = idx.norm(a, "action").astype(np.float32)
        rec["action_is_pad"] = pad
        prev = np.asarray(idx.lowdim[e["offset"] + t - 1, :D], np.float32)
        rec["prev_action"] = idx.norm(prev, "action").astype(np.float32)
        tg = seg["end"] - 1 if ds.rng.random() < 0.25 else t + int(ds.rng.uniform(0, 4.0) * idx.fps)
        rec["goal_images"] = ds._images(e, min(max(tg, t), e["length"] - 1))
        meta = dict(speed=e["speed"], quality=e["quality"], mistake=seg["mistake"])
        rec["prompt"] = pi07_prompt(e["instruction"], seg["text"], meta)
        rec["prompt_nosub"] = pi07_prompt(e["instruction"], None, meta)
        # 같은 태스크의 다른 subtask 문장 (없으면 다른 태스크에서)
        vocab = sorted({g["text"] for ep in ds.eps if ep["task"] == e["task"] for g in ep["segments"]}
                       - {seg["text"]})
        if not vocab:
            vocab = sorted({g["text"] for ep in ds.eps for g in ep["segments"]} - {seg["text"]})
        rec["prompt_wrong"] = pi07_prompt(e["instruction"], vocab[ds.rng.randrange(len(vocab))], meta)
        rec["task"] = e["task"]
        samples.append(rec)
    ds.reader.close()
    print(f"표본 {len(samples)}개 준비", flush=True)

    from models import LLPolicy
    policy = LLPolicy(device)
    sd = torch.load(os.path.join(args.ckpt, "trainable.pt"), map_location="cpu", weights_only=True)
    pmap = dict(policy.named_parameters())   # 저장은 LLPolicy 기준 이름이다 (train.save_ckpt 와 같게)
    for n, v in sd.items():
        pmap[n].data.copy_(v.to(pmap[n].device, pmap[n].dtype))
    del sd
    policy.eval()
    torch.cuda.empty_cache()

    acc = {k: [] for k in ("model", "model_goal", "hold", "zero", "median_k", "mean_k", "best_k")}
    per_step = {k: [] for k in ("model", "hold", "median_k")}
    sens = {k: [] for k in ("noise", "subtask", "nosub", "goal")}
    scale = []
    corr = 0.0
    n_corr = 0
    jitter = {k: [] for k in ("model", "median_k", "demo")}
    amp = {k: [] for k in ("model", "median_k", "demo")}
    sweep = [int(x) for x in args.steps_sweep.split(",") if x]
    acc_sweep = {st: [] for st in sweep}

    for i in range(0, len(samples), args.bs):
        b = collate(samples[i:i + args.bs])
        gt = b["actions"].to(device)
        w = (~b["action_is_pad"].to(device)).float()[..., None]
        B = gt.shape[0]
        g = torch.Generator(device=device).manual_seed(1234 + i)
        nA = torch.randn(B, H, 32, generator=g, device=device)
        nB = torch.randn(B, H, 32, generator=g, device=device)
        goal = b["goal_images"]

        with torch.no_grad():
            pA = sample_with_noise(policy, b, nA)
            pB = sample_with_noise(policy, b, nB)
            pW = sample_with_noise(policy, b, nA, prompts=[s for s in b["prompt_wrong"]])
            pN = sample_with_noise(policy, b, nA, prompts=[s for s in b["prompt_nosub"]])
            pG = sample_with_noise(policy, b, nA, goal=goal)

        # K 표본 — flow 표본의 분산과 중심을 가른다. L1 을 최소화하는 집계는 중앙값이다.
        with torch.no_grad():
            ks = torch.stack([pA, pB] + [sample_with_noise(policy, b, torch.randn(B, H, 32, generator=g, device=device))
                                         for _ in range(args.k - 2)])
        median_k = ks.median(0).values
        mean_k = ks.mean(0)
        best_k = ks.gather(0, (ks - gt[None]).abs().argmin(0, keepdim=True))[0]

        hold = b["prev_action"].to(device)[:, None, :].expand(-1, H, -1)
        zero = torch.zeros_like(gt)
        for name, p in (("model", pA), ("model_goal", pG), ("hold", hold), ("zero", zero),
                        ("median_k", median_k), ("mean_k", mean_k), ("best_k", best_k)):
            acc[name].append(((p - gt).abs() * w).cpu().numpy())
        for name, p in (("model", pA), ("hold", hold), ("median_k", median_k)):
            per_step[name].append((((p - gt).abs() * w).mean(-1)).cpu().numpy())
        sens["noise"].append(((pA - pB).abs() * w).cpu().numpy())
        sens["subtask"].append(((pA - pW).abs() * w).cpu().numpy())
        sens["nosub"].append(((pA - pN).abs() * w).cpu().numpy())
        sens["goal"].append(((pA - pG).abs() * w).cpu().numpy())
        # 출력의 떨림이 입력 잡음의 잔재인가 — (표본 − 8표본 중앙값) 와 시작 잡음의 상관
        r = (pA - median_k)[..., :D]
        z = nA[..., :D]
        rc = r - r.mean(dim=(1, 2), keepdim=True)
        zc = z - z.mean(dim=(1, 2), keepdim=True)
        corr += ((rc * zc).sum(dim=(1, 2)) / (rc.pow(2).sum(dim=(1, 2)).sqrt()
                                              * zc.pow(2).sum(dim=(1, 2)).sqrt()).clamp_min(1e-9)).sum().item()
        n_corr += rc.shape[0]
        scale.append(((gt - hold).abs() * w).cpu().numpy())  # 데모가 50스텝 동안 실제로 움직인 양
        for name, p in (("model", pA), ("median_k", median_k), ("demo", gt)):
            jitter[name].append(flip_rate(p).cpu().numpy())
            amp[name].append(step_delta(p).cpu().numpy())
        for st in sweep:                      # 디노이즈 스텝 수를 바꾼 단일 표본
            with torch.no_grad():
                ps = sample_with_noise(policy, b, nA, steps=st)
            acc_sweep[st].append(((ps - gt).abs() * w).cpu().numpy())
        print(f"  {i + B}/{len(samples)}", flush=True)

    def reduce(chunks):
        return np.concatenate(chunks, 0)       # [N, H, D] (패딩 자리는 0)

    masks = np.stack([~s["action_is_pad"] for s in samples])[..., None].astype(np.float32)  # [N,H,1]
    denom = masks.sum()

    out = {"ckpt": args.ckpt, "n": len(samples), "groups": {}, "overall": {}, "per_step": {}, "sensitivity": {}}
    for name, chunks in acc.items():
        x = reduce(chunks)
        out["overall"][name] = float(x.sum() / (denom * D))
        out["groups"][name] = {gname: float(x[..., a:bnd].sum() / (denom * (bnd - a))) for gname, a, bnd in GROUPS}
    for name, chunks in per_step.items():
        x = np.concatenate(chunks, 0)          # [N, H]
        m = masks[..., 0]
        out["per_step"][name] = (x.sum(0) / np.maximum(m.sum(0), 1e-6)).tolist()
    for name, chunks in sens.items():
        x = reduce(chunks)
        out["sensitivity"][name] = {"overall": float(x.sum() / (denom * D)),
                                    **{gname: float(x[..., a:bnd].sum() / (denom * (bnd - a)))
                                       for gname, a, bnd in GROUPS}}
    out["jitter"] = {}                      # 청크 안에서 방향이 바뀌는 비율 — 롤아웃의 떨림과 같은 척도
    for name, chunks in jitter.items():
        x = np.concatenate(chunks, 0)       # [N, D]
        out["jitter"][name] = {gname: float(x[..., a:bnd].mean()) for gname, a, bnd in GROUPS}
    out["step_delta"] = {}                  # 청크 안 스텝당 |증분| — 떨림의 진폭
    for name, chunks in amp.items():
        x = np.concatenate(chunks, 0)
        out["step_delta"][name] = {gname: float(x[..., a:bnd].mean()) for gname, a, bnd in GROUPS}
    out["noise_residual_corr"] = corr / max(n_corr, 1)
    out["steps_sweep"] = {}
    for st, chunks in acc_sweep.items():
        x = reduce(chunks)
        out["steps_sweep"][str(st)] = float(x.sum() / (denom * D))
    x = reduce(scale)
    out["demo_motion"] = {"overall": float(x.sum() / (denom * D)),
                          **{gname: float(x[..., a:bnd].sum() / (denom * (bnd - a))) for gname, a, bnd in GROUPS}}

    run_dir = os.path.dirname(os.path.dirname(args.ckpt))
    os.makedirs(os.path.join(run_dir, "analysis"), exist_ok=True)
    p = os.path.join(run_dir, "analysis", f"ll_probe_{os.path.basename(args.ckpt.rstrip('/'))}.json")
    json.dump(out, open(p, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out["overall"], indent=1))
    print(json.dumps(out["groups"]["model"], indent=1, ensure_ascii=False))
    print(json.dumps({k: v["overall"] for k, v in out["sensitivity"].items()}, indent=1))
    print("→", p)


if __name__ == "__main__":
    main()
