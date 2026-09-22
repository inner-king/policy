#!/usr/bin/env python3
"""LL(저수준 정책) 그림 — 구조도 · 진단 · 롤아웃 액션.

    vla/.venv/bin/python src/analysis/plot_pi07_ll.py

산출: docs/figures/pi07/06_ll_architecture.png · 07_ll_diagnosis.png · 08_ll_rollout.png
입력: runs/train/pi07ll-pi05base-30k/analysis/ll_probe.json (probe_pi07_ll.py),
      runs/eval/<thawing>/analysis/rollout_actions.json (probe_pi07_rollout.py) + actions-*/ep*.npy, data/pi07.
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
import numpy as np  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "pi07"))
OUT = REPO / "docs" / "figures" / "pi07"
LL = REPO / "runs" / "train" / "pi07ll-pi05base-30k"
EV = REPO / "runs" / "eval" / "pi07ll-pi05base-30k-at30k-thawing_frozen_food-ll-hl"
GROUPS = ["베이스", "토르소", "왼팔", "왼그리퍼", "오른팔", "오른그리퍼"]


def box(ax, x, y, w, h, text, fc, fontsize=8.5, ec="none", tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.01",
                                fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=tc)


def arrow(ax, p0, p1, color="black", style="-|>", lw=1.2, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --- prefix 토큰 띠
    ax.text(0.01, 0.955, "prefix — VLM 이 한 번에 읽는다 (동결, bf16, no_grad → KV 캐시만 넘김)",
            fontsize=10, weight="bold")
    y = 0.80
    h = 0.10
    segs = [(0.01, 0.13, "머리 카메라\n256 토큰", "#cfe3f7"),
            (0.145, 0.13, "왼 손목\n256 토큰", "#cfe3f7"),
            (0.28, 0.13, "오른 손목\n256 토큰", "#cfe3f7"),
            (0.415, 0.30, "텍스트 ≤170 토큰\nTask: … Subtask: … Speed: … Quality: … Mistake: …\nControl Mode: joint. State: <256구간 25개>; Action:", "#ffe6c7"),
            (0.725, 0.265, "서브골 3뷰 768 토큰\n배치의 25% 에만", "#d8f0d8")]
    for x, w, t, c in segs:
        box(ax, x, y, w, h, t, c, fontsize=8)
    ax.annotate("", xy=(0.715, y - 0.03), xytext=(0.01, y - 0.03),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1))
    ax.text(0.36, y - 0.065, "블록 안에서 양방향 (π0.5 사전학습과 같은 배치)", fontsize=8, color="gray", ha="center")
    ax.text(0.857, y - 0.05, "서브골은 앞을 전부 보지만 앞은 서브골을 못 본다",
            fontsize=8, color="tab:green", ha="center")

    # --- VLM
    box(ax, 0.01, 0.60, 0.70, 0.11, "PaliGemma 3B — SigLIP 400M + Gemma 2B  (동결)", "#b9d4ef", fontsize=10)
    box(ax, 0.725, 0.60, 0.265, 0.11, "KV 캐시\n(18층 × key/value)", "#e8e8e8", fontsize=9)
    arrow(ax, (0.36, 0.79), (0.36, 0.715))
    arrow(ax, (0.857, 0.79), (0.857, 0.715), color="tab:green")
    arrow(ax, (0.71, 0.655), (0.723, 0.655))

    # --- expert
    box(ax, 0.01, 0.37, 0.21, 0.13, "잡음 섞인 액션 청크\nx_t : 50 스텝 × 32 차원", "#f1e2f7", fontsize=9)
    box(ax, 0.25, 0.33, 0.46, 0.20,
        "action expert — Gemma 300M 폭 축소판  (학습 대상 430M)\n"
        "cross-attention 으로 prefix 의 KV 만 읽는다 · 자기들끼리 양방향\n"
        "시간 t 는 adaRMSNorm 으로 각 층에 주입", "#f7cfcf", fontsize=9)
    box(ax, 0.74, 0.37, 0.25, 0.13, "속도장 v(x_t, t)\n50 × 32", "#f1e2f7", fontsize=9)
    arrow(ax, (0.857, 0.595), (0.857, 0.505), color="gray")
    arrow(ax, (0.22, 0.435), (0.25, 0.435))
    arrow(ax, (0.71, 0.435), (0.74, 0.435))

    # --- 학습/추론
    box(ax, 0.01, 0.135, 0.47, 0.15,
        "학습:  x_t = t·ε + (1−t)·a,  목표 u = ε − a\n"
        "손실 = ‖v − u‖²  (23차원만, 패딩 9차원 제외)\n"
        "t ~ π0.5 의 beta 분포 · 유효 배치 32 · 30,000 스텝", "#fff3cc", fontsize=9)
    box(ax, 0.51, 0.135, 0.48, 0.15,
        "추론:  x ~ N(0, I) 에서 오일러 5 스텝 (t: 1 → 0)\n"
        "50 스텝 청크를 받아 앞 25 스텝만 실행하고 다시 추론\n"
        "정규화 해제 = q01/q99 역변환 → 23차원 관절 목표·베이스 속도", "#fff3cc", fontsize=9)
    ax.text(0.5, 0.055, "π0.5 base(openpi → LeRobot 포트)의 구조 그대로. π0.7 에서 가져온 것은 프롬프트 구성·드롭아웃·서브골 블록·5스텝 추론이다.",
            fontsize=9, ha="center", color="#444444")
    fig.tight_layout()
    fig.savefig(OUT / "06_ll_architecture.png", dpi=110)
    print("06_ll_architecture.png")


def fig_diagnosis():
    p = json.load(open(LL / "analysis" / "ll_probe.json"))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

    x = np.arange(len(GROUPS))
    w = 0.27
    for k, (name, lab, c) in enumerate((("model", "모델 (1 표본)", "tab:blue"),
                                        ("median_k", "모델 (8 표본 중앙값)", "tab:cyan"),
                                        ("hold", "직전 액션 반복", "gray"))):
        ax[0].bar(x + (k - 1) * w, [p["groups"][name][g] for g in GROUPS], w, label=lab, color=c)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(GROUPS, fontsize=8)
    ax[0].set_ylabel("액션 L1 (정규화, 50스텝 청크)")
    ax[0].set_title("(a) 관절군별 오차 — 자명한 기준선보다 높다")
    ax[0].legend(fontsize=8)

    s = p["sensitivity"]
    names = [("noise", "잡음만 교체\n(바닥값)"), ("subtask", "다른 subtask\n문장"), ("nosub", "subtask 제거"),
             ("goal", "서브골 이미지\n추가")]
    ax[1].bar(range(len(names)), [s[k]["overall"] for k, _ in names],
              color=["gray", "tab:red", "tab:orange", "tab:green"])
    ax[1].axhline(p["demo_motion"]["overall"], ls="--", c="k", lw=1)
    ax[1].text(0.02, p["demo_motion"]["overall"] * 1.02,
               f"참고: 데모가 50스텝 동안 실제로 움직인 양 {p['demo_motion']['overall']:.3f}", fontsize=8)
    ax[1].set_xticks(range(len(names)))
    ax[1].set_xticklabels([n for _, n in names], fontsize=8)
    ax[1].set_ylabel("출력이 움직인 양 L1")
    ax[1].set_title("(b) 조건 민감도 — 지시보다 잡음이 더 크게 흔든다")

    labs = ["1 표본", "8 중앙값", "8 평균", "8 최선\n(오라클)", "직전 액션\n반복"]
    keys = ["model", "median_k", "mean_k", "best_k", "hold"]
    cols = ["tab:blue", "tab:cyan", "tab:cyan", "tab:green", "gray"]
    ax[2].bar(range(len(keys)), [p["overall"][k] for k in keys], color=cols)
    sw = p.get("steps_sweep", {})
    ks = sorted(sw, key=int)
    ax[2].plot([len(keys) + i for i in range(len(ks))], [sw[k] for k in ks], "o-", color="tab:red")
    for i, k in enumerate(ks):
        ax[2].annotate(f"{k}스텝", (len(keys) + i, sw[k]), fontsize=8, color="tab:red",
                       xytext=(0, 7), textcoords="offset points", ha="center")
    ax[2].set_xticks(list(range(len(keys))) + [len(keys) + i for i in range(len(ks))])
    ax[2].set_xticklabels(labs + [f"{k}\n스텝" for k in ks], fontsize=8)
    ax[2].text(len(keys) + (len(ks) - 1) / 2, ax[2].get_ylim()[1] * 0.02, "디노이즈 스텝 수 (1 표본)",
               fontsize=8, ha="center", color="tab:red")
    ax[2].set_ylabel("액션 L1")
    ax[2].set_title("(c) 좋은 모드는 있다 — 한 표본을 그대로 쓰는 게 문제")
    for a in ax:
        a.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "07_ll_diagnosis.png", dpi=110)
    print("07_ll_diagnosis.png")


def fig_rollout():
    from dataset import B1KIndex
    r = json.load(open(EV / "analysis" / "rollout_actions.json"))
    probe = json.load(open(LL / "analysis" / "ll_probe.json"))
    idx = B1KIndex()
    e = [x for x in idx.episodes if x["task"] == "thawing_frozen_food"][0]
    demo = np.asarray(idx.lowdim[e["offset"]:e["offset"] + e["length"], :23], np.float32)
    roll = np.load(EV / "actions-local" / "ep00.npy")

    def busiest(x, j, n):
        """가장 많이 움직이는 n 프레임 창의 시작 위치 — 둘 다 '일하는 중'인 구간을 비교한다."""
        d = np.abs(np.diff(x[:, j]))
        c = np.convolve(d, np.ones(n), "valid")
        return int(c.argmax())

    fig = plt.figure(figsize=(15, 6.4))
    gs = fig.add_gridspec(2, 3)
    n = 90                                   # 3초
    t = np.arange(n) / 30
    for row, (j, name, center) in enumerate(((15, "오른팔 1번 관절", True), (22, "오른 그리퍼", False))):
        a = fig.add_subplot(gs[row, 0])
        r0, d0 = busiest(roll, j, n), busiest(demo, j, n)
        rv, dv = roll[r0:r0 + n, j].copy(), demo[d0:d0 + n, j].copy()
        if center:
            rv -= rv.mean()
            dv -= dv.mean()
        a.plot(t, rv, color="tab:blue", lw=1.1, label="롤아웃 (정책 명령)")
        a.plot(t, dv, color="tab:green", lw=1.1, label="사람 데모")
        for k in range(1, int(n / 25) + 1):
            a.axvline(k * 25 / 30, color="tab:red", lw=0.5, alpha=0.6)
        suffix = " (각자 평균을 뺀 값)" if center else ""
        a.set_title(f"({'ad'[row]}) {name} — 가장 많이 움직인 3초{suffix}", fontsize=9.5)
        a.set_xlabel("초 · 세로선 = 25스텝 재추론", fontsize=8)
        a.tick_params(labelsize=8)
        a.grid(alpha=0.3)
        a.legend(fontsize=8)

    x = np.arange(len(GROUPS))
    w = 0.27
    b = fig.add_subplot(gs[:, 1])
    b.barh(x + w, [r["rollout"]["sign_flip_rate"][g] for g in GROUPS], w, color="tab:blue", label="롤아웃 액션")
    b.barh(x, [probe["jitter"]["model"][g] for g in GROUPS], w, color="tab:cyan", label="예측 청크 (검증 관측)")
    b.barh(x - w, [r["demo"]["sign_flip_rate"][g] for g in GROUPS], w, color="tab:green", label="사람 데모")
    b.set_yticks(x)
    b.set_yticklabels(GROUPS, fontsize=9)
    b.set_xlabel("증분의 방향이 바뀌는 비율")
    b.set_title("(b) 떨림 — 거의 매 스텝 방향이 뒤집힌다")
    b.legend(fontsize=8)
    b.grid(axis="x", alpha=0.3)

    c = fig.add_subplot(gs[:, 2])
    c.barh(x + w / 2, [r["rollout"]["chunk_boundary_delta"][g] for g in GROUPS], w, color="tab:red",
           label="재추론 경계 (25스텝마다)")
    c.barh(x - w / 2, [r["rollout"]["in_chunk_delta"][g] for g in GROUPS], w, color="tab:gray", label="청크 내부")
    c.set_yticks(x)
    c.set_yticklabels(GROUPS, fontsize=9)
    c.set_xlabel("스텝당 액션 변화량 (원단위)")
    c.set_title("(c) 이어 붙인 자리에서 튄다 (RTC 미구현)")
    c.legend(fontsize=8)
    c.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "08_ll_rollout.png", dpi=110)
    print("08_ll_rollout.png")




def fig_scaling():
    """10k/20k/30k 체크포인트의 멱법칙과 200k 외삽 — 200k 재학습의 근거."""
    import numpy as np
    S = (10000, 20000, 30000)
    D = [json.load(open(LL / "analysis" / f"ll_probe_step-{s:06d}.json")) for s in S]
    x = np.array(S, float)
    xs = np.logspace(np.log10(8000), np.log10(220000), 50)

    def fit(y):
        e = np.log(y[-1] / y[0]) / np.log(x[-1] / x[0])
        a = y[-1] / x[-1] ** e
        return e, a * xs ** e

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    series = [("전체 23차원", [d["overall"]["model"] for d in D], "tab:blue",
               D[0]["overall"]["hold"]),
              ("왼팔", [d["groups"]["model"]["왼팔"] for d in D], "tab:green",
               D[0]["groups"]["hold"]["왼팔"]),
              ("오른팔", [d["groups"]["model"]["오른팔"] for d in D], "tab:red",
               D[0]["groups"]["hold"]["오른팔"])]
    for name, y, c, base in series:
        y = np.array(y)
        e, line = fit(y)
        ax[0].plot(x, y, "o", color=c)
        ax[0].plot(xs, line, "-", color=c, lw=1.2, label=f"{name} (지수 {e:+.2f}, 200k {line[-1]:.4f})")
        ax[0].axhline(base, ls=":", color=c, lw=1, alpha=0.7)
    ax[0].axvline(200000, color="gray", lw=1)
    ax[0].annotate("200k", xy=(200000, 0.02), xytext=(3, 0), textcoords="offset points",
                   fontsize=8, color="gray", rotation=90, va="bottom")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("학습 스텝 (로그)")
    ax[0].set_ylabel("액션 L1 (정규화)")
    ax[0].set_title("(a) 포화 없는 멱법칙 — 점선은 '직전 액션 반복' 기준선")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")

    sub = np.array([d["sensitivity"]["subtask"]["overall"] for d in D])
    noi = np.array([d["sensitivity"]["noise"]["overall"] for d in D])
    es, ls_ = fit(sub)
    en, ln = fit(noi)
    ax[1].plot(x, sub, "o", color="tab:red")
    ax[1].plot(xs, ls_, "-", color="tab:red", lw=1.2, label=f"subtask 문장의 영향 (지수 {es:+.2f})")
    ax[1].plot(x, noi, "o", color="gray")
    ax[1].plot(xs, ln, "-", color="gray", lw=1.2, label=f"초기 잡음의 영향 (지수 {en:+.2f})")
    cross = xs[np.argmin(np.abs(ls_ - ln))]
    ax[1].axvline(cross, color="k", ls="--", lw=1)
    ax[1].text(cross * 1.06, ls_.min() * 1.1, f"교차 ≈ {cross / 1000:.0f}k\n여기서부터 지시가\n잡음보다 세진다", fontsize=8)
    ax[1].axvline(200000, color="gray", lw=1)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("학습 스텝 (로그)")
    ax[1].set_ylabel("출력이 움직인 양 L1")
    ax[1].set_title("(b) annotation 이 출력을 지배하기 시작하는 지점")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "09_ll_scaling.png", dpi=110)
    print("09_ll_scaling.png")


if __name__ == "__main__":
    plt.rcParams["font.family"] = ["Noto Sans CJK KR", "Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    OUT.mkdir(parents=True, exist_ok=True)
    fig_architecture()
    fig_diagnosis()
    fig_rollout()
    fig_scaling()
