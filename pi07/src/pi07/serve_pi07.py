#!/usr/bin/env python3
"""π0.7 런타임 (Algorithm 1) — BEHAVIOR-1K 평가용 websocket 정책 서버.

    ℓ̂ ← HL(o_t, ℓ, 이력)                    hl_period 스텝마다 (기본 30 = 1초)
    g* ← WM(o_t, ℓ̂)                          ℓ̂ 가 바뀌거나 4초가 지나면 (π0.7 VII, Δ=4 s)
    a_{t:t+50} ← LL(o_t, ℓ, ℓ̂, g*, m, c)     Ĥ 스텝마다 (기본 25), denoising 5 스텝
    m = {speed: 태스크별 학습 에피소드 길이 15 백분위, quality: 5, mistake: false},  c = joint

π0.7 은 HL·WM 을 별도 스레드에서 비동기로 돌린다. BEHAVIOR 평가 하네스는 정책 응답을 기다리는 동안
시뮬레이터를 멈추므로 여기서는 동기로 돌려도 결과가 같다 (지연이 점수에 들어가지 않는다).
⚠️ websocket ping 기본 20초 — WM 한 번이 그보다 길면 연결이 끊길 수 있어 ping 을 끈다.

메모리 (24 GB 한 장, 챌린지 제출 조건): HL 과 LL 이 PaliGemma 하나를 나눠 쓴다.
HL 은 LoRA 를 켠 상태, LL 은 끈 상태로 같은 가중치를 쓴다.

평가 프로토콜은 src/vla/serve_smolvla.py 와 같다 (openpi 유래 msgpack + ndarray 확장).

    vla/.venv/bin/python src/pi07/serve_pi07.py \\
        --ll runs/train/pi07ll-pi05base-30k/checkpoints/step-030000 \\
        --hl runs/train/pi07hl-pi05base-8k/checkpoints/step-008000 \\
        [--wm runs/train/pi07wm-flux2kleinbase-10k/checkpoints/step-010000] --port 8000
    ... --self-test          # 더미 관측 + 검증 에피소드 관측으로 한 바퀴 돌고 끝
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import B1KIndex, hl_prompt, pi07_prompt  # noqa: E402
from models import HLPolicy, LLPolicy, add_lora, set_lora  # noqa: E402

CAPSTONE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(CAPSTONE, "src", "vla"))
from serve_smolvla import Packer, _health_check, make_dummy_obs, unpackb  # noqa: E402

# lerobot 가 먼저 로깅을 설정해 두므로 force 가 없으면 INFO 가 버려진다 (2026-09-19 스모크 평가에서 확인)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
log = logging.getLogger("serve_pi07")

CAMERA_MARKERS = (("zed_link", 0), ("left_realsense_link", 1), ("right_realsense_link", 2))
STATE_SLICES = [(0, 3), (53, 57), (3, 10), (24, 26), (28, 35), (49, 51)]
PROPRIO_DIM = 61  # STATE_SLICES 는 61차원 proprio 기준 인덱스다 (src/vla/serve_smolvla.py 의 STATE_DIM 과 같은 값)
FALLBACK_TASK_ID = 93  # clean_a_keyboard — 관측의 task_id 가 없거나 index.json 에 없을 때 쓰는 문장 (경고를 남긴다)


def load_trainable(module: torch.nn.Module, ckpt_dir: str, rename=None):
    sd = torch.load(os.path.join(ckpt_dir, "trainable.pt"), map_location="cpu", weights_only=True)
    params = dict(module.named_parameters())
    n = 0
    for k, v in sd.items():
        k2 = rename(k) if rename else k
        if k2 not in params:
            raise KeyError(f"{k} → {k2} 가 모델에 없음")
        params[k2].data.copy_(v.to(params[k2].device, params[k2].dtype))
        n += 1
    return n


class Pi07Runtime:
    def __init__(self, ll_ckpt: str, hl_ckpt: str | None, wm_ckpt: str | None, device: str = "cuda:0",
                 hl_period: int = 30, exec_steps: int = 25, wm_period_s: float = 4.0, use_subgoal: bool = True,
                 meta_cfg: float = 0.0, fixed_subtask: str | None = None, max_history: int = 8,
                 wm_steps: int = 28, wm_cfg: float = 4.0, trace_dir: str | None = None):
        self.device = torch.device(device)
        self.idx = B1KIndex()
        self.ll = LLPolicy(self.device)
        n = load_trainable(self.ll, ll_ckpt)
        log.info(f"LL: {ll_ckpt} ({n} tensors)")
        for p in self.ll.parameters():
            p.requires_grad_(False)
        self.ll.eval()
        self.hl = None
        vlm = self.ll.model.paligemma_with_expert.paligemma
        if hl_ckpt:
            # 같은 PaliGemma 에 LoRA 를 붙이고 HL 가중치를 싣는다 (이름: vlm.… → 공유 모듈 기준)
            self.hl = HLPolicy(self.device, vlm=vlm, grad_ckpt=False)
            n = load_trainable(self.hl, hl_ckpt)
            log.info(f"HL: {hl_ckpt} ({n} tensors)")
            self.hl.eval()
            set_lora(vlm, False)
        self.wm = None
        if wm_ckpt:
            from world_model import WorldModel
            self.wm = WorldModel(self.device, grad_ckpt=False)
            n = load_trainable(self.wm, wm_ckpt)
            log.info(f"WM: {wm_ckpt} ({n} tensors)")
            self.wm.eval()
        self.hl_period = hl_period
        self.exec_steps = exec_steps
        self.wm_period = int(round(wm_period_s * self.idx.fps))
        self.use_subgoal = use_subgoal and self.wm is not None
        self.meta_cfg = meta_cfg
        self.fixed_subtask = fixed_subtask
        self.max_history = max_history
        self.use_hl = self.hl is not None  # 연결(포트)별로 끌 수 있다 — 플랫 기준선 조건
        self.wm_steps = wm_steps
        self.wm_cfg = wm_cfg
        # 에피소드별 결정 기록 (HL 출력·WM 호출·LL 프롬프트) — 평가 뒤 전환 시점 분석용
        self.trace_dir = trace_dir
        self.n_episode = -1
        self._trace_f = None
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
        self.task_by_id = {}
        vocab: dict[str, set] = {}
        for e in self.idx.episodes:
            self.task_by_id[e["task_index"]] = (e["task"], e["instruction"])
            if e["split"] == "train":
                vocab.setdefault(e["task"], set()).update(g["text"] for g in e["segments"])
        # HL 출력을 태스크의 학습 subtask 문장 집합으로 제한한다 (트라이 제약 디코딩)
        self.hl_trie = {t: self.hl.build_trie(sorted(v)) for t, v in vocab.items()} if self.hl else {}
        self.reset()

    # 연결마다 따로 가지는 에피소드 상태. 모델 가중치는 하나를 공유하고, act() 앞뒤로 이 상태를 바꿔 끼운다.
    # 서버의 이벤트 루프가 act() 를 한 번에 하나만 돌리므로 (동기 호출) 교체가 안전하다.
    STATE_KEYS = ("t", "queue", "subtask", "history", "subgoal", "subgoal_t", "trace", "_trace_f", "n_episode",
                  "trace_dir", "use_hl")

    def get_state(self) -> dict:
        return {k: getattr(self, k) for k in self.STATE_KEYS}

    def set_state(self, st: dict):
        for k, v in st.items():
            setattr(self, k, v)

    def new_state(self, trace_dir: str | None, use_hl: bool = True) -> dict:
        saved = self.get_state()
        self.use_hl = use_hl and self.hl is not None
        self._trace_f = None
        self.n_episode = -1
        self.trace_dir = trace_dir
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
        self.reset()
        st = self.get_state()
        self.set_state(saved)
        return st

    def reset(self):
        # 기록 파일은 리셋 뒤 첫 관측에서 연다 — 하네스가 첫 에피소드 전에 리셋을 보내든 안 보내든 번호가 맞게
        if self._trace_f is not None:
            self._trace_f.close()
            self._trace_f = None
        self.t = 0
        self.queue: list[np.ndarray] = []
        self.subtask: str | None = self.fixed_subtask
        self.history: list[str] = []
        self.subgoal: torch.Tensor | None = None
        self.subgoal_t = -10 ** 9
        self.trace: list[dict] = []
        self._warned_task_ids: set = set()

    # --- 관측 변환 ---
    @staticmethod
    def _img(arr: np.ndarray, size: int = 224) -> torch.Tensor:
        arr = np.asarray(arr)
        if arr.ndim == 1 and arr.dtype == np.uint8:
            # 원격 시뮬레이터(src/pi07/eval_remote.py)가 224 로 줄이고 JPEG 으로 보낸 것 — 회선이 2 MB/s 라서
            import io
            from PIL import Image
            arr = np.asarray(Image.open(io.BytesIO(arr.tobytes())).convert("RGB"))
        t = torch.from_numpy(np.ascontiguousarray(arr[..., :3]))
        if t.dtype != torch.uint8:
            t = (t.float() * (255.0 if float(t.max()) <= 1.5 else 1.0)).clamp(0, 255).to(torch.uint8)
        if t.shape[0] != size or t.shape[1] != size:
            x = t.permute(2, 0, 1)[None].float()
            x = F.interpolate(x, size=(size, size), mode="area")
            t = x[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8)
        return t

    def _parse(self, obs: dict):
        views = [None, None, None]
        state = None
        task_id = None
        for k, v in obs.items():
            if not isinstance(k, str):
                continue
            if k.endswith("::rgb"):
                for marker, i in CAMERA_MARKERS:
                    if marker in k:
                        views[i] = self._img(v)
            elif k.endswith("::proprio"):
                state = np.asarray(v, dtype=np.float32).reshape(-1)
                assert state.shape[0] == PROPRIO_DIM, f"proprio {state.shape} != {PROPRIO_DIM}"
            elif k == "task_id":
                task_id = int(np.asarray(v).reshape(-1)[0])
        assert all(x is not None for x in views) and state is not None, list(obs)[:8]
        s25 = np.concatenate([state[a:b] for a, b in STATE_SLICES])
        return torch.stack(views)[None], s25, task_id

    # --- 한 스텝 ---
    @torch.no_grad()
    def act(self, obs: dict) -> np.ndarray:
        t0 = time.monotonic()
        images, s25, task_id = self._parse(obs)
        if self.trace_dir and self._trace_f is None:
            self.n_episode += 1
            self._trace_f = open(os.path.join(self.trace_dir, f"ep{self.n_episode:02d}.jsonl"), "w")
        if task_id not in self.task_by_id:
            if task_id not in self._warned_task_ids:  # 에피소드당 한 번 (reset 에서 비운다)
                self._warned_task_ids.add(task_id)
                log.warning(f"task_id={task_id!r} 가 index.json 에 없다 (관측에 task_id 키가 없으면 None) — "
                            f"{FALLBACK_TASK_ID} ({self.task_by_id[FALLBACK_TASK_ID][0]}) 의 문장으로 대체한다")
            task_id = FALLBACK_TASK_ID
        task, instr = self.task_by_id[task_id]
        rec = dict(t=self.t)
        if self.use_hl and self.fixed_subtask is None and self.t % self.hl_period == 0:
            set_lora(self.hl.vlm, True)
            try:
                # 이력 = 지금까지 낸 지시 (현재 유효한 것 포함) — dataset.HLDataset 과 같은 정의
                prompt = hl_prompt(instr, self.history[-self.max_history:])
                new = self.hl.generate(images, [prompt], allowed=[self.hl_trie.get(task)])[0]
            finally:
                set_lora(self.hl.vlm, False)  # generate 가 예외로 끝나도 LL 이 LoRA 켜진 VLM 을 쓰지 않게
            if not self.history or self.history[-1] != new:
                self.history.append(new)
            if new != self.subtask:
                log.info(f"t={self.t} subtask: {self.subtask!r} → {new!r}")
                self.subtask = new
                self.subgoal_t = -10 ** 9  # 서브골 즉시 갱신
                self.queue = []            # 새 지시로 청크를 다시 뽑는다
            rec["hl"] = new
        if self.use_subgoal and self.subtask and (self.t - self.subgoal_t >= self.wm_period):
            self.subgoal = self.wm.generate(images, [instr], [self.subtask], steps=self.wm_steps, cfg=self.wm_cfg)
            self.subgoal_t = self.t
            rec["wm"] = True
        if not self.queue:
            meta = dict(speed=self.idx.runtime_speed.get(task, 5000), quality=5, mistake=False)
            prompt = pi07_prompt(instr, self.subtask, meta)
            st = self.idx.norm(s25, "state")[None]
            goal = self.subgoal if self.use_subgoal else None
            chunk = self._sample(images, prompt, st, goal, instr)
            a = self.idx.unnorm_action(chunk[0].float().cpu().numpy())
            self.queue = list(a[:self.exec_steps])
            rec["ll_prompt"] = prompt
            # LL 이 받은 state 25차원, 정규화 전 값 — 평가 시점 proprio 분포를 학습 분포와 대조하기 위해
            rec["state"] = [round(float(x), 6) for x in s25]
        self.t += 1
        action = self.queue.pop(0).astype(np.float32)
        # 응답의 server_timing.infer_ms 와 trace 에 같은 값이 간다 (Server._handler 가 이 값을 싣는다)
        self.infer_ms = (time.monotonic() - t0) * 1000
        if len(rec) > 1:
            rec["infer_ms"] = round(self.infer_ms, 1)
            rec["wall"] = round(time.time(), 3)  # 벽시계, unix epoch 초 (UTC)
            self.trace.append(rec)
            if self._trace_f is not None:
                self._trace_f.write(json.dumps(rec) + "\n")
                self._trace_f.flush()
        return action

    def _sample(self, images, prompt, st, goal, instr):
        if self.meta_cfg <= 0:
            return self.ll.sample_actions(images, [prompt], st, goal)
        # π0.7 VII: 메타데이터에 CFG. v = v_c + β (v_c − v_u), 무조건 = 메타데이터를 뺀 프롬프트
        uprompt = pi07_prompt(instr, self.subtask, None)
        pad_c, kv_c = self.ll.encode_prefix(images, [prompt], st, goal)
        pad_u, kv_u = self.ll.encode_prefix(images, [uprompt], st, goal)
        import copy
        x = torch.randn(1, 50, 32, device=self.device)
        steps = 5
        for i in range(steps):
            t = torch.full((1,), 1.0 - i / steps, device=self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vc = self.ll.expert_velocity(pad_c, copy.deepcopy(kv_c), x, t).float()
                vu = self.ll.expert_velocity(pad_u, copy.deepcopy(kv_u), x, t).float()
            x = x - (vc + self.meta_cfg * (vc - vu)) / steps
        return x[..., :23]


class Server:
    """포트마다 조건(HL 켬/끔)과 기록 위치를 따로 둔다. 모델은 하나를 공유한다."""

    def __init__(self, rt: Pi07Runtime, host: str, ports: list[tuple[int, bool, str | None]], metadata: dict):
        self.rt, self.host, self.ports, self.metadata = rt, host, ports, metadata
        self.n_conn = 0

    def serve_forever(self):
        asyncio.run(self._run())

    async def _run(self):
        import functools as ft
        import websockets.asyncio.server as _server
        servers = []
        for port, use_hl, trace in self.ports:
            log.info(f"websocket 서버 시작: {self.host}:{port} use_hl={use_hl} trace={trace}")
            servers.append(await _server.serve(ft.partial(self._handler, use_hl=use_hl, trace=trace), self.host, port,
                                               compression=None, max_size=None, process_request=_health_check,
                                               ping_interval=None))
        await asyncio.gather(*(s_.serve_forever() for s_ in servers))

    async def _handler(self, ws, use_hl: bool = True, trace: str | None = None):
        import websockets
        packer = Packer()
        await ws.send(packer.pack({**self.metadata, "use_hl": use_hl}))
        self.n_conn += 1
        peer = str(ws.remote_address[0]) if ws.remote_address else "unknown"
        st = self.rt.new_state(os.path.join(trace, f"conn{self.n_conn:02d}_{peer}") if trace else None, use_hl=use_hl)
        log.info(f"연결 {self.n_conn}: {peer} use_hl={use_hl}")
        while True:
            try:
                msg = unpackb(await ws.recv())
                saved = self.rt.get_state()
                self.rt.set_state(st)
                try:
                    if "reset" in msg:
                        self.rt.reset()
                        continue
                    a = self.rt.act(msg)
                finally:
                    st = self.rt.get_state()
                    self.rt.set_state(saved)
                await ws.send(packer.pack({"action": a, "server_timing": {"infer_ms": self.rt.infer_ms}}))
            except websockets.ConnectionClosed:
                break
            except Exception:
                log.error(traceback.format_exc())
                await ws.close(code=1011, reason="internal error")
                raise


def self_test(rt: Pi07Runtime, n_steps: int):
    """더미 관측 한 스텝 + 검증 에피소드의 실제 관측으로 n_steps 스텝 (HL·LL·WM 전부 거친다)."""
    from frames import FrameReader
    rt.reset()
    a = rt.act(make_dummy_obs())
    assert a.shape == (23,), a.shape
    e = rt.idx.split("val")[0]
    rd = FrameReader()
    rt.reset()
    t0 = time.time()
    errs = []
    for t in range(0, n_steps):
        obs = {"task_id": np.array([e["task_index"]]),
               "robot_r1::proprio": np.zeros(61, np.float32)}
        st = rt.idx.lowdim[e["offset"] + t]
        full = np.zeros(61, np.float32)
        s25 = st[rt.idx.action_dim:]
        o = 0
        for a_, b_ in STATE_SLICES:
            full[a_:b_] = s25[o:o + b_ - a_]
            o += b_ - a_
        obs["robot_r1::proprio"] = full
        for (marker, i), v in zip(CAMERA_MARKERS, ("head", "left_wrist", "right_wrist")):
            vid = e["videos"][v]
            obs[f"robot_r1::robot_r1:{marker}:Camera:0::rgb"] = rd.read(vid["path"], vid["from_ts"] + t / 30)
        a = rt.act(obs)
        errs.append(np.abs(a - st[:rt.idx.action_dim]).mean())
    dt = time.time() - t0
    gt = e["segments"][rt.idx.segment_at(e, 0)]["text"]
    print(f"[self-test] {e['task']} ep {e['episode_index']}: {n_steps} steps {dt:.1f}s "
          f"({dt / n_steps * 1000:.0f} ms/step), |a - a_demo| mean {np.mean(errs):.3f} (raw units)")
    print(f"[self-test] HL subtask {rt.subtask!r} (정답 t=0: {gt!r}), history {rt.history}")
    print(f"[self-test] GPU peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    wm_calls = sum(1 for r in rt.trace if r.get("wm"))
    print(f"[self-test] HL 질의 {sum(1 for r in rt.trace if 'hl' in r)}회, WM 생성 {wm_calls}회, "
          f"LL 청크 {sum(1 for r in rt.trace if 'll_prompt' in r)}회")
    for r in rt.trace[:6]:
        print("   ", json.dumps(r)[:300])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ll", required=True)
    ap.add_argument("--hl", default=None)
    ap.add_argument("--wm", default=None)
    ap.add_argument("--no-subgoal", action="store_true", help="WM 을 싣더라도 서브골을 LL 에 넣지 않는다")
    ap.add_argument("--fixed-subtask", default=None, help="HL 대신 고정 subtask (사람 코칭 대용)")
    ap.add_argument("--hl-period", type=int, default=30)
    ap.add_argument("--exec-steps", type=int, default=25, help="π0.7: Ĥ ∈ {15, 25}")
    ap.add_argument("--meta-cfg", type=float, default=0.0, help="π0.7: β ∈ {1.3, 1.7, 2.2}, 0 = 끔")
    ap.add_argument("--wm-steps", type=int, default=28, help="서브골 생성 denoising 스텝 (3090 에서 스텝당 약 0.35 s, CFG 포함)")
    ap.add_argument("--wm-cfg", type=float, default=4.0)
    ap.add_argument("--trace-dir", default=None, help="에피소드별 HL/WM/LL 결정 JSONL 저장 위치")
    ap.add_argument("--flat-port", type=int, default=None, help="이 포트로 온 연결은 HL 없이 (태스크 문장만) — 플랫 기준선")
    ap.add_argument("--flat-trace-dir", default=None)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--self-test", type=int, default=0, help="검증 에피소드 관측으로 N 스텝 돌고 종료")
    args = ap.parse_args()
    rt = Pi07Runtime(args.ll, args.hl, args.wm, device=args.device, hl_period=args.hl_period,
                     exec_steps=args.exec_steps, use_subgoal=not args.no_subgoal, meta_cfg=args.meta_cfg,
                     fixed_subtask=args.fixed_subtask, wm_steps=args.wm_steps, wm_cfg=args.wm_cfg,
                     trace_dir=args.trace_dir)
    if args.self_test:
        self_test(rt, args.self_test)
        return
    ports = [(args.port, True, args.trace_dir)]
    if args.flat_port:
        ports.append((args.flat_port, False, args.flat_trace_dir))
    Server(rt, args.host, ports, metadata={"policy": "pi07_b1k", "hl": args.hl, "wm": args.wm,
                                           "ll": args.ll}).serve_forever()


if __name__ == "__main__":
    main()
