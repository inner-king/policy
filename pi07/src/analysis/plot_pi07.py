#!/usr/bin/env python3
"""π0.7 재현 요약 그림 — 학습 곡선 · HL 타임라인 · 평가 프레임 · 월드 모델 표본.

    groot/.venv/bin/python src/analysis/plot_pi07.py            # docs/figures/pi07/*.png 전부 다시 생성

입력은 전부 산출물이다: runs/train/<id>/logs/metrics.jsonl, runs/train/<id>/val/*.png,
runs/eval/<id>/{videos,trace}/…, data/pi07/index.json. 손으로 적은 수치는 없다.
"""
from __future__ import annotations

import glob
import json
import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"  # 한글 라벨용 (NanumGothic 없음)


def font(size: int):
    try:
        return ImageFont.truetype(CJK_FONT, size)
    except OSError:
        return ImageFont.load_default()

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "figures" / "pi07"
TRAIN = REPO / "runs" / "train"
EVAL = REPO / "runs" / "eval"
THAW_A = EVAL / "pi07ll-pi05base-30k-at30k-thawing_frozen_food-ll-hl"
FPS = 30


def metrics(run: str, key: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for line in open(TRAIN / run / "logs" / "metrics.jsonl"):
        d = json.loads(line)
        if key in d:
            xs.append(d["step"])
            ys.append(d[key])
    return xs, ys


def fig_training():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    # (a) HL 검증
    s, ex = metrics("pi07hl-pi05base-8k", "val_exact")
    _, sw = metrics("pi07hl-pi05base-8k", "val_exact_switch")
    _, st = metrics("pi07hl-pi05base-8k", "val_exact_stay")
    _, bl = metrics("pi07hl-pi05base-8k", "val_stay_baseline")
    ax[0].plot(s, ex, "o-", label="전체 정답률")
    ax[0].plot(s, st, "^-", color="tab:green", label="유지해야 할 때")
    ax[0].plot(s, sw, "s-", color="tab:red", label="전환해야 할 때")
    ax[0].axhline(bl[0], ls="--", c="gray", label=f"직전 지시 유지 기준선 {bl[0]:.3f}")
    ax[0].set_title("(a) 고수준 정책 검증 (384 표본)")
    ax[0].set_xlabel("스텝")
    ax[0].set_ylim(0, 1)
    ax[0].legend(fontsize=8)

    # (b) LL 검증 L1 (30k 뒤에 genmix 5k 를 이어 붙인다)
    s1, g1 = metrics("pi07ll-pi05base-30k", "val_goal_l1")
    _, n1 = metrics("pi07ll-pi05base-30k", "val_nogoal_l1")
    s2, g2 = metrics("pi07ll-pi05base-30k-genmix-5k", "val_goal_l1")
    _, n2 = metrics("pi07ll-pi05base-30k-genmix-5k", "val_nogoal_l1")
    s2 = [x + s1[-1] for x in s2]
    ax[1].plot(s1, n1, "o-", color="tab:blue", label="서브골 없음")
    ax[1].plot(s1, g1, "o-", color="tab:orange", label="서브골 있음")
    ax[1].plot(s2, n2, "o--", color="tab:blue")
    ax[1].plot(s2, g2, "o--", color="tab:orange")
    ax[1].axvline(s1[-1], c="gray", lw=1)
    ax[1].text(s1[-1] * 1.005, max(n1) * 0.98, "생성 서브골 혼합 (3단계)", fontsize=8, color="gray")
    ax[1].set_title("(b) 저수준 VLA 검증 — 데모 대비 액션 L1")
    ax[1].set_xlabel("스텝")
    ax[1].legend(fontsize=8)

    # (c) WM PSNR
    s3, pp = metrics("pi07wm-flux2kleinbase-6k", "val_psnr_pred")
    _, pc = metrics("pi07wm-flux2kleinbase-6k", "val_psnr_copy")
    ax[2].plot(s3, pp, "o-", label="생성한 서브골")
    ax[2].axhline(pc[0], ls="--", c="gray", label=f"현재 관측 복사 {pc[0]:.2f} dB")
    ax[2].set_title("(c) 월드 모델 검증 — 정답 대비 PSNR")
    ax[2].set_xlabel("스텝")
    ax[2].set_ylabel("dB")
    ax[2].legend(fontsize=8)

    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "01_training.png", dpi=110)
    print("01_training.png")


def hl_timeline():
    """조건 A 10 에피소드의 HL 지시 구간 + 데모 중앙값 경계."""
    rows = []
    for f in sorted(glob.glob(str(THAW_A / "trace" / "conn*" / "ep*.jsonl"))):
        seq = []
        for line in open(f):
            r = json.loads(line)
            if "hl" in r and (not seq or seq[-1][1] != r["hl"]):
                seq.append((r["t"] / FPS, r["hl"]))
        rows.append((os.path.basename(os.path.dirname(f))[:6] + "/" + os.path.basename(f)[:4], seq))
    idx = json.load(open(REPO / "data" / "pi07" / "index.json"))
    segs = [e["segments"] for e in idx["episodes"] if e["task"] == "thawing_frozen_food"]
    d1 = np.median([s[0]["end"] / FPS for s in segs])
    d2 = np.median([s[1]["end"] / FPS for s in segs])

    colors = {"move to the fridge": "tab:blue", "open the fridge": "tab:orange"}
    fig, ax = plt.subplots(figsize=(11, 4.4))
    end = 12345 / FPS
    for i, (name, seq) in enumerate(rows):
        for j, (t0, lab) in enumerate(seq):
            t1 = seq[j + 1][0] if j + 1 < len(seq) else end
            ax.barh(i, t1 - t0, left=t0, height=0.62, color=colors.get(lab, "tab:gray"),
                    edgecolor="white", linewidth=0.6)
    ax.axvline(d1, color="k", ls="--", lw=1)
    ax.axvline(d2, color="k", ls=":", lw=1)
    ax.text(d1 + 2, len(rows) - 0.3, f"데모: 냉장고 도착 {d1:.0f}초", fontsize=8)
    ax.text(d2 + 2, len(rows) - 1.1, f"데모: 문 열기 완료 {d2:.0f}초", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"inst {301 + i}" for i in range(len(rows))], fontsize=8)
    ax.set_xlabel("초 (제한 시간 411.5초)")
    ax.set_xlim(0, end)
    ax.set_title("고수준 정책이 낸 지시 — 파랑 move to the fridge · 주황 open the fridge")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "02_hl_timeline.png", dpi=110)
    print("02_hl_timeline.png")


def frames(video: str, want: list[int]) -> dict[int, np.ndarray]:
    import av
    c = av.open(video)
    s = c.streams.video[0]
    out = {}
    for i, f in enumerate(c.decode(s)):
        if i in want:
            out[i] = f.to_ndarray(format="rgb24")
        if i > max(want):
            break
    c.close()
    return out


def grid(imgs: list[np.ndarray], labels: list[str], path: pathlib.Path, per_row: int, scale: float = 1.0):
    h, w = imgs[0].shape[:2]
    rows = (len(imgs) + per_row - 1) // per_row
    canvas = Image.new("RGB", (w * per_row, (h + 18) * rows), "white")
    d = ImageDraw.Draw(canvas)
    f = font(14)
    for k, (im, lab) in enumerate(zip(imgs, labels)):
        r, c = divmod(k, per_row)
        canvas.paste(Image.fromarray(im), (c * w, r * (h + 18) + 18))
        d.text((c * w + 4, r * (h + 18) + 2), lab, fill="black", font=f)
    if scale != 1.0:
        canvas = canvas.resize((int(canvas.width * scale), int(canvas.height * scale)))
    canvas.save(path)
    print(path.name)


def fig_rollouts():
    # 309 — 문이 실제로 열린 구간 (머리 카메라 = 영상 우측 448 크롭)
    v = str(THAW_A / "videos" / "thawing_frozen_food_309_0.mp4")
    want = [10500, 10650, 10740, 11000, 11400, 11700, 11790, 12100]
    fr = frames(v, want)
    grid([fr[w][:, 448:] for w in want], [f"{w / FPS:.0f}초" for w in want],
         OUT / "03_fridge_open_309.png", 4, scale=0.75)
    # 301 — 전형적 실패: 냉장고 앞에서 팔만 움직인다
    v = str(THAW_A / "videos" / "thawing_frozen_food_301_0.mp4")
    want = [600, 3000, 5400, 7200, 9000, 10800, 11700, 12300]
    fr = frames(v, want)
    grid([fr[w][:, 448:] for w in want], [f"{w / FPS:.0f}초" for w in want],
         OUT / "04_typical_failure_301.png", 4, scale=0.75)


def fig_wm():
    src = Image.open(TRAIN / "pi07wm-flux2kleinbase-6k" / "val" / "step-006000.png")
    w, h = src.size           # 8행 × (현재3 | 생성3 | 정답3)
    row = h // 8
    crop = src.crop((0, 0, w, row * 3))
    crop = crop.resize((int(w * 0.62), int(row * 3 * 0.62)))
    canvas = Image.new("RGB", (crop.width, crop.height + 20), "white")
    canvas.paste(crop, (0, 20))
    d = ImageDraw.Draw(canvas)
    f = font(15)
    for i, lab in enumerate(("현재 관측 3뷰", "월드 모델이 생성한 서브골", "정답 (구간 끝 프레임)")):
        d.text((i * crop.width // 3 + 6, 2), lab, fill="black", font=f)
    canvas.save(OUT / "05_world_model.png")
    print("05_world_model.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Noto Sans CJK KR", "Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig_training()
    hl_timeline()
    fig_rollouts()
    fig_wm()
