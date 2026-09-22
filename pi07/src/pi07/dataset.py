"""π0.7 재현 — B1K 학습 표본 생성기 (HL / LL / WM 공용).

build_index.py 가 만든 $DATA/pi07/{index.json, lowdim.npy, stats.json} 을 읽는다.
에폭 개념 없이 무한 무작위 표본을 낸다 (IterableDataset). 시드 = (base_seed, rank, worker, 재개 스텝).

π0.7 논문(arXiv:2604.15483)에서 가져온 표본 규칙:
- 프롬프트:  Task: … Subtask: … Speed: … Quality: … Mistake: … Control Mode: joint.   (V-E)
- 드롭아웃:  서브골 이미지는 배치의 25% 에만. 서브골이 있는 예시에서 subtask 30% 제거.
            메타데이터 통째로 15%, 항목별 5%. control mode 는 드롭하지 않음.       (V-E)
- 서브골 시점: 25% 세그먼트 끝 프레임, 75% 현재~4초 뒤 균등.                        (VI-C)
- speed = 에피소드 길이를 500 스텝 단위로 반올림.                                  (V-C)
- HL 입력 = 관측 + 태스크 + 지난 subtask 이력, 출력 = 지금 할 subtask.            (V-A)

메모리: lowdim 은 memmap(파일 기반)이라 워커 간 복사되지 않는다. 비디오는 frames.FrameReader 가
열린 파일 수를 제한한다.
"""
from __future__ import annotations

import json
import math
import os
import random
from bisect import bisect_right

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from frames import FrameReader

CAPSTONE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PI07_DATA = os.path.join(CAPSTONE, "data", "pi07")
VIEWS = ("head", "left_wrist", "right_wrist")


class B1KIndex:
    def __init__(self, root: str = PI07_DATA):
        meta = json.load(open(os.path.join(root, "index.json")))
        self.fps = meta["fps"]
        self.action_dim = meta["action_dim"]
        self.state_dim = meta["state_dim"]
        self.episodes = meta["episodes"]
        self.lowdim = np.load(os.path.join(root, "lowdim.npy"), mmap_mode="r")
        self.stats = json.load(open(os.path.join(root, "stats.json")))
        for e in self.episodes:
            e["_starts"] = [g["start"] for g in e["segments"]]
        # 태스크별 speed 15 백분위 — 런타임 프롬프트 값 (π0.7 VII)
        by_task = {}
        for e in self.episodes:
            if e["split"] == "train":
                by_task.setdefault(e["task"], []).append(e["length"])
        self.runtime_speed = {t: int(round(np.percentile(v, 15) / 500) * 500) for t, v in by_task.items()}

    def split(self, name: str) -> list[dict]:
        return [e for e in self.episodes if e["split"] == name]

    # --- 세그먼트 조회 ---
    @staticmethod
    def segment_at(e: dict, t: int) -> int:
        """프레임 t 에 활성인 세그먼트 번호. 구간 사이 빈틈(프레임의 1.3%)이면 다음 세그먼트."""
        i = bisect_right(e["_starts"], t) - 1
        segs = e["segments"]
        if i < 0:
            return 0
        if t < segs[i]["end"]:
            return i
        return min(i + 1, len(segs) - 1)

    @staticmethod
    def history(e: dict, seg_idx: int, max_len: int) -> list[str]:
        """세그먼트 0..seg_idx 의 문장들 (seg_idx 포함). 연속 중복(같은 스킬을 두 조각으로 나눈 주석)은 하나로.

        HL 의 이력은 "지금까지 내린 지시" 다 — 현재 유효한 지시(마지막 원소)를 포함한다. 런타임에서 HL 은
        자기가 직전에 낸 지시를 이력 끝에 두고, 그 지시를 유지할지 다음으로 넘어갈지를 이미지로 판단한다.
        """
        if seg_idx < 0:
            return []
        out = []
        for g in e["segments"][:seg_idx + 1]:
            if not out or out[-1] != g["text"]:
                out.append(g["text"])
        return out[-max_len:] if max_len > 0 else []

    # --- 저차원 ---
    def norm(self, x: np.ndarray, key: str) -> np.ndarray:
        """π0.5 방식 분위수 정규화: q01→-1, q99→+1."""
        q01 = np.asarray(self.stats[key]["q01"], np.float32)
        q99 = np.asarray(self.stats[key]["q99"], np.float32)
        return np.clip((x - q01) / np.maximum(q99 - q01, 1e-6) * 2.0 - 1.0, -1.5, 1.5)

    def unnorm_action(self, a: np.ndarray) -> np.ndarray:
        q01 = np.asarray(self.stats["action"]["q01"], np.float32)
        q99 = np.asarray(self.stats["action"]["q99"], np.float32)
        return (a + 1.0) / 2.0 * (q99 - q01) + q01

    def state(self, e: dict, t: int) -> np.ndarray:
        return np.asarray(self.lowdim[e["offset"] + t, self.action_dim:], np.float32)

    def action_chunk(self, e: dict, t: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        """a[t : t+H] 와 패딩 마스크. 에피소드 끝을 넘는 자리는 마지막 액션 반복 + is_pad."""
        end = min(t + horizon, e["length"])
        a = np.asarray(self.lowdim[e["offset"] + t:e["offset"] + end, :self.action_dim], np.float32)
        pad = np.zeros(horizon, dtype=bool)
        if len(a) < horizon:
            pad[len(a):] = True
            a = np.concatenate([a, np.repeat(a[-1:], horizon - len(a), axis=0)], axis=0)
        return a, pad


def worker_seed(base: int, rank: int, step: int) -> int:
    wi = get_worker_info()
    wid = wi.id if wi is not None else 0
    return (base * 1_000_003 + rank * 10_007 + wid * 101 + step) % (2 ** 31)


class _Base(IterableDataset):
    def __init__(self, split: str, image_size: int = 224, seed: int = 0, rank: int = 0, start_step: int = 0,
                 index: B1KIndex | None = None):
        super().__init__()
        self.split_name = split
        self.image_size = image_size
        self.seed = seed
        self.rank = rank
        self.start_step = start_step
        self._index = index

    def _setup(self):
        self.idx = self._index or B1KIndex()
        self.eps = self.idx.split(self.split_name)
        w = np.array([e["length"] for e in self.eps], dtype=np.float64)
        self.ep_p = w / w.sum()  # 프레임 균등 표본
        self.rng = random.Random(worker_seed(self.seed, self.rank, self.start_step))
        self.nrng = np.random.default_rng(worker_seed(self.seed + 7, self.rank, self.start_step))
        self.reader = FrameReader(fps=self.idx.fps)

    def _pick_episode(self) -> dict:
        return self.eps[int(self.nrng.choice(len(self.eps), p=self.ep_p))]

    def _images(self, e: dict, t: int) -> np.ndarray:
        """세 뷰 [3, H, W, 3] uint8."""
        out = []
        for v in VIEWS:
            vid = e["videos"][v]
            t = min(max(t, 0), e["length"] - 1)
            out.append(self.reader.read(vid["path"], vid["from_ts"] + t / self.idx.fps, size=self.image_size))
        return np.stack(out)


def hl_prompt(instruction: str, history: list[str] | None) -> str:
    """HL 프롬프트. history=None 이면 이력 드롭아웃 표기."""
    if history is None:
        h = "unknown"
    else:
        h = "; ".join(history) if history else "none"
    return f"Task: {instruction} Subtask history: {h}. Current subtask:"


class HLDataset(_Base):
    """High-level policy 표본: (3뷰 관측, 태스크, 지시 이력) → 지금 유효해야 할 subtask 문장.

    이력 = 직전 HL 질의 시점(t − Δ)까지 내려진 지시들 (그 시점의 지시 포함). Δ ~ U[0.5, 1.5]초 —
    런타임 질의 주기 1초 근처. 그래서
      - 세그먼트 중간: 이력의 마지막 = 정답 (유지)
      - 경계 직후 Δ 안: 이력의 마지막 = 이전 세그먼트, 정답 = 다음 세그먼트 (전환)
    둘을 가르는 정보는 이미지에만 있다.

    프레임의 절반은 에피소드 전체에서 균등, 절반은 세그먼트 경계 ±boundary_s 초 안에서 뽑는다.
    """

    def __init__(self, split: str, max_history: int = 8, history_dropout: float = 0.1,
                 boundary_frac: float = 0.5, boundary_s: float = 1.5, query_dt: tuple[float, float] = (0.5, 1.5),
                 **kw):
        super().__init__(split, **kw)
        self.max_history = max_history
        self.history_dropout = history_dropout
        self.boundary_frac = boundary_frac
        self.boundary_s = boundary_s
        self.query_dt = query_dt

    def make(self, e: dict, t: int, dt_frames: int, drop_history: bool = False) -> dict:
        si = self.idx.segment_at(e, t)
        tp = t - dt_frames
        hist = self.idx.history(e, self.idx.segment_at(e, tp), self.max_history) if tp >= 0 else []
        g = e["segments"][si]
        return dict(images=self._images(e, t), prompt=hl_prompt(e["instruction"], None if drop_history else hist),
                    target=g["text"], task=e["task"], episode=e["episode_index"], t=t,
                    switch=bool(hist and hist[-1] != g["text"]))

    def sample(self) -> dict:
        e = self._pick_episode()
        L = e["length"]
        if self.rng.random() < self.boundary_frac and len(e["segments"]) > 1:
            g = e["segments"][self.rng.randrange(1, len(e["segments"]))]
            t = g["start"] + int(self.rng.uniform(-self.boundary_s, self.boundary_s) * self.idx.fps)
        else:
            t = self.rng.randrange(L)
        t = min(max(t, 0), L - 1)
        dt = int(self.rng.uniform(*self.query_dt) * self.idx.fps)
        return self.make(e, t, dt, drop_history=self.rng.random() < self.history_dropout)

    def __iter__(self):
        self._setup()
        while True:
            yield self.sample()


def pi07_prompt(instruction: str, subtask: str | None, meta: dict | None) -> str:
    """π0.7 V-E 프롬프트 (이미지·고유수용 토큰 제외 텍스트 부분). meta 의 빠진 키는 생략된다."""
    parts = [f"Task: {instruction.rstrip('.')}."]
    if subtask:
        parts.append(f"Subtask: {subtask}.")
    if meta:
        if "speed" in meta:
            parts.append(f"Speed: {meta['speed']}.")
        if "quality" in meta:
            parts.append(f"Quality: {meta['quality']}.")
        if "mistake" in meta:
            parts.append(f"Mistake: {'true' if meta['mistake'] else 'false'}.")
    parts.append("Control Mode: joint.")
    return " ".join(parts)


class LLDataset(_Base):
    """Low-level VLA 표본 — π0.7 컨텍스트 + 드롭아웃.

    with_goal=True 인 인스턴스는 서브골 3뷰를 항상 붙이고(배치의 25%를 이 스트림에서 채운다),
    False 인 인스턴스는 서브골을 붙이지 않는다. 배치를 두 스트림으로 나누는 이유: 서브골 토큰
    768개를 서브골 없는 75% 에 패딩으로 붙이면 연산이 낭비된다.
    """

    def __init__(self, split: str, with_goal: bool, horizon: int = 50, goal_end_p: float = 0.25,
                 goal_max_s: float = 4.0, p_subtask_drop_goal: float = 0.3, p_subtask_drop_nogoal: float = 0.1,
                 p_meta_drop: float = 0.15, p_meta_item_drop: float = 0.05, gen_dir: str | None = None,
                 p_gen: float = 0.5, **kw):
        super().__init__(split, **kw)
        # π0.7 VI-C: 서브골 예시의 p_gen 만큼은 월드 모델 생성 이미지 (src/pi07/gen_subgoals.py 산출물)
        self.gen_dir = gen_dir
        self.p_gen = p_gen
        self.with_goal = with_goal
        self.horizon = horizon
        self.goal_end_p = goal_end_p
        self.goal_max_s = goal_max_s
        self.p_subtask_drop = p_subtask_drop_goal if with_goal else p_subtask_drop_nogoal
        self.p_meta_drop = p_meta_drop
        self.p_meta_item_drop = p_meta_item_drop

    def _setup(self):
        super()._setup()
        self.gen = None
        if self.with_goal and self.gen_dir:
            meta = json.load(open(os.path.join(self.gen_dir, "samples.json")))
            done = np.load(os.path.join(self.gen_dir, "done.npy"))
            self.gen = dict(samples=meta["samples"], ok=np.nonzero(done)[0],
                            images=np.load(os.path.join(self.gen_dir, "images.npy"), mmap_mode="r"))
            self.by_ep = {e["episode_index"]: e for e in self.eps}

    def sample(self) -> dict:
        gen_i = None
        if self.gen is not None and len(self.gen["ok"]) and self.rng.random() < self.p_gen:
            gen_i = int(self.gen["ok"][self.rng.randrange(len(self.gen["ok"]))])
            s = self.gen["samples"][gen_i]
            e, t = self.by_ep[s["episode_index"]], s["t"]
        else:
            e = self._pick_episode()
            t = self.rng.randrange(e["length"])
        return self.make(e, t, gen_i)

    def make(self, e: dict, t: int, gen_i: int | None = None) -> dict:
        """(에피소드, t) 가 정해진 표본. 검증에서 같은 (에피소드, t) 에 서브골 유/무를 짝지을 때 직접 부른다.
        rng 호출 순서는 sample() 하나였을 때와 같다 (같은 시드면 같은 표본열)."""
        si = self.idx.segment_at(e, t)
        seg = e["segments"][si]
        subtask = None if self.rng.random() < self.p_subtask_drop else seg["text"]
        meta = None
        if self.rng.random() >= self.p_meta_drop:
            meta = {}
            for k, v in (("speed", e["speed"]), ("quality", e["quality"]), ("mistake", seg["mistake"])):
                if self.rng.random() >= self.p_meta_item_drop:
                    meta[k] = v
        out = dict(prompt=pi07_prompt(e["instruction"], subtask, meta), episode=e["episode_index"], t=t)
        out["images"] = self._images(e, t)
        if gen_i is not None:
            out["goal_images"] = np.asarray(self.gen["images"][gen_i])
        elif self.with_goal:
            if self.rng.random() < self.goal_end_p:
                tg = seg["end"] - 1
            else:
                tg = t + int(self.rng.uniform(0.0, self.goal_max_s) * self.idx.fps)
            tg = min(max(tg, t), e["length"] - 1)
            out["goal_images"] = self._images(e, tg)
        st = self.idx.norm(self.idx.state(e, t), "state")
        a, pad = self.idx.action_chunk(e, t, self.horizon)
        out["state"] = st.astype(np.float32)
        out["actions"] = self.idx.norm(a, "action").astype(np.float32)
        out["action_is_pad"] = pad
        return out

    def __iter__(self):
        self._setup()
        while True:
            yield self.sample()


class WMDataset(_Base):
    """World model 표본: (현재 3뷰, subtask, 메타) → 세그먼트 끝 3뷰 (π0.7 V-B, App. C)."""

    def __init__(self, split: str, **kw):
        super().__init__(split, **kw)

    def sample(self) -> dict:
        e = self._pick_episode()
        t = self.rng.randrange(e["length"])
        si = self.idx.segment_at(e, t)
        seg = e["segments"][si]
        tg = max(seg["end"] - 1, t)
        return dict(images=self._images(e, t), goal_images=self._images(e, tg), subtask=seg["text"],
                    instruction=e["instruction"], speed=e["speed"], task=e["task"])

    def __iter__(self):
        self._setup()
        while True:
            yield self.sample()


def collate(batch: list[dict]) -> dict:
    out = {}
    for k in batch[0]:
        v = [b[k] for b in batch]
        if isinstance(v[0], np.ndarray):
            out[k] = torch.from_numpy(np.stack(v))
        else:
            out[k] = v
    return out
