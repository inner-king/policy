"""학습된 SmolVLA 체크포인트를 BEHAVIOR-1K 평가용 websocket 정책 서버로 서빙한다.

평가 쪽(`python -m omnigibson.eval.eval --policy websocket`)이 openpi 프로토콜로
접속한다: msgpack(+ndarray 확장) / 접속 직후 metadata 1프레임 / {'reset': True} 리셋 /
obs dict → {'action': ndarray} 응답 / GET /healthz 헬스체크.

이 스크립트는 omnigibson을 import하지 않는다 (평가와 다른 venv에서 돌기 때문).
프로토콜 구현은 OmniGibson/omnigibson/eval/utils/network_utils.py(openpi 유래)에서
벤더링했다 — 와이어 포맷이 정확히 일치해야 한다.

관측 매핑 (evaluator._preprocess_obs → 우리 데이터셋 키):
    robot_r1::<sensor>::rgb (zed/left_realsense/right_realsense)
        → observation.rgb.{zed,left_realsense,right_realsense}_link_camera_0
    robot_r1::proprio (61)  → observation.state
    task 텍스트             → 학습 때와 동일한 "clean_a_keyboard"
    (cam_rel_poses, task_id는 SmolVLA가 쓰지 않으므로 버린다)

⚠️ 평가는 RGBDFullResWrapper로 돌릴 것 (head 720x720 / wrist 480x480).
   기본 DefaultWrapper는 224x224라 학습 분포(720/480→512 resize)와 어긋난다.

실행::

    vla/.venv/bin/python src/vla/serve_smolvla.py \
        --checkpoint runs/train/smolvla-100k/checkpoints/080000/pretrained_model \
        --port 8000

    # 서버 자가 검증만 (더미 관측 2스텝):
    vla/.venv/bin/python src/vla/serve_smolvla.py --checkpoint ... --self-test
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import http
import logging
import time
import traceback
from pathlib import Path

import msgpack
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("serve_smolvla")

ACTION_DIM = 23
STATE_DIM = 61
TASK_TEXT = "clean_a_keyboard"  # 학습 데이터의 task 문자열과 동일해야 한다

# 평가 obs 키 부분문자열 → 데이터셋 feature 이름
CAMERA_MAP = {
    "zed_link": "observation.rgb.zed_link_camera_0",
    "left_realsense_link": "observation.rgb.left_realsense_link_camera_0",
    "right_realsense_link": "observation.rgb.right_realsense_link_camera_0",
}


# ---------------------------------------------------------------------------
# msgpack + ndarray 확장 (network_utils.py에서 벤더링 — 와이어 포맷 유지)
# ---------------------------------------------------------------------------
def pack_data(obj):
    if isinstance(obj, torch.Tensor):
        data = obj.detach().cpu().numpy()
        return {b"__ndarray__": True, b"data": data.tobytes(), b"dtype": data.dtype.str, b"shape": data.shape}
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype: {obj.dtype}")
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def unpack_data(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=pack_data)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_data)


# ---------------------------------------------------------------------------
# SmolVLA 정책 래퍼
# ---------------------------------------------------------------------------
class B1KSmolVLAPolicy:
    """평가 obs dict → lerobot 배치 → SmolVLA select_action → 23-dim 액션."""

    def __init__(self, checkpoint: str, device: str = "cuda",
                 n_action_steps: int | None = None) -> None:
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.device = device
        logger.info(f"체크포인트 로드: {checkpoint}")
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint).to(device).eval()
        if n_action_steps is not None:
            # 청크(50)는 그대로 예측하되 앞 n개만 실행하고 재추론한다 (receding horizon).
            # 학습 기본값 50은 1.67초 개루프 실행이라, 시각 서보가 필요한 파지 구간에서
            # 반응성이 없다는 가설(analysis 2026-08-16 §7)을 검증하기 위한 플래그.
            # 공식 베이스라인 π0.5·GR00T가 공통으로 쓰는 값은 16이다.
            assert 1 <= n_action_steps <= self.policy.config.chunk_size, n_action_steps
            self.policy.config.n_action_steps = n_action_steps
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        logger.info(f"n_action_steps={self.policy.config.n_action_steps}, "
                    f"chunk_size={self.policy.config.chunk_size}")

    @staticmethod
    def _to_image_tensor(arr: np.ndarray) -> torch.Tensor:
        """(H,W,3|4) uint8/float → (1,3,H,W) float32 [0,1]"""
        t = torch.from_numpy(np.ascontiguousarray(arr))
        if t.ndim == 2:  # 안전망
            t = t.unsqueeze(-1).repeat(1, 1, 3)
        t = t[..., :3]
        if t.dtype == torch.uint8:
            t = t.float() / 255.0
        else:
            t = t.float()
            if t.max() > 1.5:  # 0-255 float로 오는 경우
                t = t / 255.0
        return t.permute(2, 0, 1).unsqueeze(0)

    def _build_batch(self, obs: dict) -> dict:
        batch: dict = {}
        for key, value in obs.items():
            if not isinstance(key, str):
                continue
            if key.endswith("::rgb"):
                for marker, feature in CAMERA_MAP.items():
                    if marker in key:
                        batch[feature] = self._to_image_tensor(np.asarray(value))
                        break
            elif key.endswith("::proprio"):
                state = torch.from_numpy(np.asarray(value, dtype=np.float32)).reshape(1, -1)
                assert state.shape[-1] == STATE_DIM, f"proprio {state.shape} != {STATE_DIM}"
                batch["observation.state"] = state
        missing = [f for f in (*CAMERA_MAP.values(), "observation.state") if f not in batch]
        assert not missing, f"관측에서 못 찾은 feature: {missing} / 받은 키: {list(obs)[:8]}"
        batch["task"] = [TASK_TEXT]
        return batch

    @torch.no_grad()
    def act(self, obs: dict) -> torch.Tensor:
        batch = self.pre(self._build_batch(obs))
        action = self.policy.select_action(batch)   # 내부 큐: chunk마다 1회 추론
        action = self.post(action)
        action = action.squeeze(0).float().cpu()
        return action[:ACTION_DIM]

    def reset(self) -> None:
        self.policy.reset()


# ---------------------------------------------------------------------------
# websocket 서버 (network_utils.WebsocketPolicyServer에서 벤더링)
# ---------------------------------------------------------------------------
class WebsocketPolicyServer:
    def __init__(self, policy, host="0.0.0.0", port=8000, metadata=None):
        self._policy, self._host, self._port = policy, host, port
        self._metadata = metadata or {}

    def serve_forever(self):
        asyncio.run(self._run())

    async def _run(self):
        import websockets.asyncio.server as _server

        logger.info(f"websocket 서버 시작: {self._host}:{self._port}")
        async with _server.serve(
            self._handler, self._host, self._port,
            compression=None, max_size=None, process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket):
        import websockets

        logger.info(f"클라이언트 접속: {websocket.remote_address}")
        packer = Packer()
        await websocket.send(packer.pack(self._metadata))
        while True:
            try:
                msg = unpackb(await websocket.recv())
                if "reset" in msg:
                    self._policy.reset()
                    continue
                t0 = time.monotonic()
                action = self._policy.act(msg)
                await websocket.send(packer.pack({
                    "action": action.cpu().numpy(),
                    "server_timing": {"infer_ms": (time.monotonic() - t0) * 1000},
                }))
            except websockets.ConnectionClosed:
                logger.info("클라이언트 접속 종료")
                break
            except Exception:
                logger.error(traceback.format_exc())
                await websocket.close(code=1011, reason="internal error")
                raise


def _health_check(connection, request):
    if getattr(request, "path", None) == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


# ---------------------------------------------------------------------------
def make_dummy_obs() -> dict:
    """evaluator._preprocess_obs가 보내는 것과 같은 모양의 더미 관측."""
    return {
        "robot_r1::robot_r1:zed_link:Camera:0::rgb": np.zeros((720, 720, 4), dtype=np.uint8),
        "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": np.zeros((480, 480, 4), dtype=np.uint8),
        "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": np.zeros((480, 480, 4), dtype=np.uint8),
        "robot_r1::proprio": np.zeros(STATE_DIM, dtype=np.float32),
        "robot_r1::cam_rel_poses": np.zeros(21, dtype=np.float32),
        "task_id": np.array([93], dtype=np.int64),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-action-steps", type=int, default=None,
                        help="추론 1회당 실행할 액션 수 (기본: 체크포인트 값 50)")
    parser.add_argument("--self-test", action="store_true", help="더미 관측 2스텝만 돌리고 종료")
    args = parser.parse_args()

    assert Path(args.checkpoint).exists(), f"체크포인트 없음: {args.checkpoint}"
    policy = B1KSmolVLAPolicy(args.checkpoint, device=args.device,
                              n_action_steps=args.n_action_steps)

    print("[self-test] 더미 관측으로 act() 2회")
    policy.reset()
    for i in range(2):
        t0 = time.monotonic()
        action = policy.act(make_dummy_obs())
        print(f"  step {i}: action shape={tuple(action.shape)} "
              f"range=[{action.min():.3f}, {action.max():.3f}] "
              f"({(time.monotonic() - t0) * 1000:.0f} ms)")
        assert action.shape == (ACTION_DIM,)
    print("[self-test] ✅ 통과")
    if args.self_test:
        return

    WebsocketPolicyServer(policy, args.host, args.port,
                          metadata={"policy": "smolvla_b1k", "task": TASK_TEXT}).serve_forever()


if __name__ == "__main__":
    main()
