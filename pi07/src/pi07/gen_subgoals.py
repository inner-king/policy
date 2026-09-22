#!/usr/bin/env python3
"""π0.7 VI-C — 월드 모델이 만든 서브골 이미지를 LL 학습용으로 미리 생성한다 (재개 가능).

    "we mitigate the train-test mismatch between real and generated images by also sampling a large number of
     subgoal images from the world model, and constructing additional training examples with these generated
     images added into the context of π0.7 instead of the real future images."

표본 = train 분할의 (에피소드, 프레임 t) 균등. 각 표본에 대해 WM(o_t, 태스크, t 의 subtask) 를 생성해
uint8 memmap [N, 3, 224, 224, 3] 에 쓴다. LLDataset(gen_dir=...) 이 이 (에피소드, t) 에서 관측·액션을 읽고
서브골 자리에 생성 이미지를 넣는다.

    python src/pi07/gen_subgoals.py --wm runs/train/pi07wm-flux2kleinbase-6k/checkpoints/step-006000 \
        --out data/pi07/gen_subgoals/wm6k --n 4000 --steps 12 --cfg 4.0 --bs 8

재개: done.npy (완료 표시) 를 보고 남은 것만 만든다. 표본 목록(samples.json)은 첫 실행 때 고정한다.
메모리: 이미지는 파일 기반 memmap 에 바로 쓴다. 메인 RSS 는 모델 로드 외에 ~1 GB.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import B1KIndex, WMDataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    idx = B1KIndex()

    sp = os.path.join(args.out, "samples.json")
    if not os.path.exists(sp):
        rng = np.random.default_rng(args.seed)
        eps = idx.split("train")
        w = np.array([e["length"] for e in eps], np.float64)
        pick = rng.choice(len(eps), size=args.n, p=w / w.sum())
        samples = [dict(episode_index=eps[i]["episode_index"], t=int(rng.integers(eps[i]["length"]))) for i in pick]
        json.dump(dict(wm=args.wm, steps=args.steps, cfg=args.cfg, samples=samples), open(sp, "w"))
    meta = json.load(open(sp))
    samples = meta["samples"]
    N = len(samples)
    ip = os.path.join(args.out, "images.npy")
    dp = os.path.join(args.out, "done.npy")
    if not os.path.exists(ip):
        np.lib.format.open_memmap(ip, mode="w+", dtype=np.uint8, shape=(N, 3, 224, 224, 3)).flush()
        np.save(dp, np.zeros(N, dtype=bool))
    imgs = np.load(ip, mmap_mode="r+")
    done = np.load(dp)
    todo = np.nonzero(~done)[0]
    print(f"{args.out}: {N - len(todo)}/{N} 완료, 남은 {len(todo)}", flush=True)
    if len(todo) == 0:
        open(os.path.join(args.out, "COMPLETED"), "w").close()
        return

    from world_model import WorldModel
    dev = torch.device("cuda:0")
    wm = WorldModel(dev, grad_ckpt=False)
    sd = torch.load(os.path.join(args.wm, "trainable.pt"), map_location="cpu", weights_only=True)
    params = dict(wm.named_parameters())
    for k, v in sd.items():
        params[k].data.copy_(v.to(params[k].device, params[k].dtype))
    wm.eval()

    by_ep = {e["episode_index"]: e for e in idx.episodes}
    ds = WMDataset("train", index=idx)
    ds._setup()
    t0 = time.time()
    for bi in range(0, len(todo), args.bs):
        ii = todo[bi:bi + args.bs]
        obs, instr, sub = [], [], []
        for i in ii:
            s = samples[i]
            e = by_ep[s["episode_index"]]
            obs.append(ds._images(e, s["t"]))
            instr.append(e["instruction"])
            sub.append(e["segments"][idx.segment_at(e, s["t"])]["text"])
        g = wm.generate(torch.from_numpy(np.stack(obs)), instr, sub, steps=args.steps, cfg=args.cfg,
                        seed=args.seed + int(ii[0]))
        imgs[ii] = g.cpu().numpy()
        imgs.flush()
        done[ii] = True
        np.save(dp + ".tmp.npy", done)
        os.replace(dp + ".tmp.npy", dp)
        n_done = int(done.sum())
        el = time.time() - t0
        rate = (bi + len(ii)) / el
        print(f"[{time.strftime('%m-%d %H:%M:%S')}] {n_done}/{N} {rate:.2f} img/s "
              f"eta {(len(todo) - bi - len(ii)) / max(rate, 1e-6) / 3600:.1f}h", flush=True)
    open(os.path.join(args.out, "COMPLETED"), "w").close()


if __name__ == "__main__":
    main()
