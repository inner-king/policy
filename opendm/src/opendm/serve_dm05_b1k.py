"""DM0.5 HTTP 서비스(OpenDM 공식 추론)를 BEHAVIOR-1K 평가용 websocket 정책 서버로 잇는 브리지.

양 끝은 공식 코드 그대로다. 모델 쪽은 `opendm/script/dm05_launcher.sh --task inference` 가 띄우는
`POST /v1/infer`, 시뮬 쪽은 `python -m omnigibson.eval.eval --policy websocket`(또는
src/pi07/replay_oracle.py). 둘 다 상대를 모르므로 여기서 프로토콜과 로봇 차원만 바꾼다.
와이어 포맷은 serve_smolvla.py 와 같다 (network_utils.py 에서 벤더링한 msgpack+ndarray, 접속 직후
metadata 1프레임, {'reset': True}, obs → {'action': (23,)}, GET /healthz).

⚠️ DM0.5 base 는 R1Pro 를 학습한 적이 없다 (DOS W1·Aloha, 팔 6관절). 이 브리지로 확인하는 것은
배관 — 요청이 오가고 로봇이 명령대로 움직이는가 — 이지 과제 수행이 아니다. 매핑은 임의로 정했다:

    state14 = [왼팔 qpos 0:6, 왼그리퍼, 오른팔 qpos 0:6, 오른그리퍼]   R1Pro 7번째 관절은 안 보낸다
    그리퍼   = 손가락 qpos 평균 / 0.05 × 0.071 (DOS W1 q99). DOS W1 은 값이 클수록 열린다고 가정 (미확인)
    액션 23  = base 속도 0 · torso 는 청크 시점 값 고정 · 팔 0:6 은 DM0.5 절대 목표 · 7번째 관절 고정 ·
               그리퍼 = clip(g/0.071, 0, 1)·2−1  (R1Pro smooth 그리퍼: −1 닫힘, +1 완전 개방)

DM0.5 응답은 서비스가 이미 state 를 더한 절대 관절 목표다 (`ActionAbsolute`, 그리퍼는 절대값 그대로).
청크 50 중 앞 --exec-steps 개만 실행하고 다시 요청한다.

실행 (DM0.5 서비스가 7891 에 떠 있어야 한다, 평가는 b1k_wrappers.RGBFullResWrapper — head 720 / wrist 480)::

    vla/.venv/bin/python src/opendm/serve_dm05_b1k.py --port 8040 --prompt "..." --trace <out>/trace.jsonl

    # 자가 검증만 (더미 관측 2스텝, 실제 DM0.5 서비스에 요청한다):
    vla/.venv/bin/python src/opendm/serve_dm05_b1k.py --self-test
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import functools
import http
import io
import json
import logging
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

import msgpack
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("serve_dm05")

ACTION_DIM = 23
STATE_DIM = 61
DM_ACTION_DIM = 14
DM_CHUNK = 50

# R1Pro proprio 인덱스 (evaluator._preprocess_obs 기준)
ARM_LEFT_QPOS = slice(3, 10)      # 7관절, 마지막(6)은 DM0.5에 안 보낸다
GRIPPER_LEFT_QPOS = slice(24, 26)
ARM_RIGHT_QPOS = slice(28, 35)
GRIPPER_RIGHT_QPOS = slice(49, 51)
TRUNK_QPOS = slice(53, 57)

R1PRO_FINGER_MAX = 0.05   # R1Pro 그리퍼 핑거 최대 개방 (m)
DM_GRIP_MAX = 0.071       # DM0.5 DOS W1 그리퍼 state/action q99 (norm_stats.json)

# 카메라 키 접미사 → DM0.5 images 슬롯
CAMERA_SUFFIXES = {
    "zed_link:Camera:0::rgb": "1",
    "left_realsense_link:Camera:0::rgb": "2",
    "right_realsense_link:Camera:0::rgb": "3",
}


# ---------------------------------------------------------------------------
# msgpack + ndarray 확장 (network_utils.py에서 벤더링 — 와이어 포맷 유지)
# ---------------------------------------------------------------------------
def pack_data(obj):
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
# 관측 → DM0.5 요청 변환
# ---------------------------------------------------------------------------
def _find_camera(obs: dict, suffix: str) -> np.ndarray | None:
    """키가 suffix로 끝나는 rgb 관측을 찾는다. 정확히 하나여야 한다."""
    hits = [v for k, v in obs.items()
            if isinstance(k, str) and k.endswith(suffix)]
    if not hits:
        return None
    if len(hits) > 1:
        raise ValueError(f"카메라 키가 여러 개 매칭됨: {suffix} ({len(hits)}개)")
    return np.asarray(hits[0])


def _encode_jpeg(arr: np.ndarray, quality: int) -> str:
    """(H,W,3|4) uint8 → 알파 버리고 RGB JPEG → base64 ascii."""
    a = np.asarray(arr)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    a = a[..., :3]
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(a)).save(
        buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _gripper_norm(qpos: np.ndarray) -> float:
    """R1Pro 그리퍼 qpos(2,) → DM0.5 명령 [0, DM_GRIP_MAX]."""
    v = float(np.mean(np.asarray(qpos, dtype=np.float64)))
    return float(np.clip(v / R1PRO_FINGER_MAX, 0.0, 1.0) * DM_GRIP_MAX)


def build_state14(proprio: np.ndarray) -> np.ndarray:
    """proprio(61,) → DM0.5 state(14,) float32."""
    p = np.asarray(proprio, dtype=np.float32).reshape(-1)
    assert p.shape[-1] == STATE_DIM, f"proprio {p.shape} != {STATE_DIM}"
    s = np.empty(DM_ACTION_DIM, dtype=np.float32)
    s[0:6] = p[ARM_LEFT_QPOS][:6]
    s[6] = _gripper_norm(p[GRIPPER_LEFT_QPOS])
    s[7:13] = p[ARM_RIGHT_QPOS][:6]
    s[13] = _gripper_norm(p[GRIPPER_RIGHT_QPOS])
    return s


def dm14_to_r1pro23(dm: np.ndarray, proprio: np.ndarray) -> np.ndarray:
    """DM0.5 액션(14,) → R1Pro 23차원 액션. 청크 뽑은 시점의 proprio 기준."""
    d = np.asarray(dm, dtype=np.float32).reshape(-1)
    assert d.shape[-1] == DM_ACTION_DIM, f"dm action {d.shape} != {DM_ACTION_DIM}"
    p = np.asarray(proprio, dtype=np.float32).reshape(-1)
    assert p.shape[-1] == STATE_DIM, f"proprio {p.shape} != {STATE_DIM}"

    out = np.zeros(ACTION_DIM, dtype=np.float32)
    # base 0:3 = 속도 0
    out[3:7] = p[TRUNK_QPOS]                       # torso 절대 (고정)
    out[7:13] = d[0:6]                             # left_arm 6
    out[13] = p[ARM_LEFT_QPOS][6]                  # left_arm 마지막 관절 (고정)
    out[14] = np.clip(d[6] / DM_GRIP_MAX, 0.0, 1.0) * 2.0 - 1.0
    out[15:21] = d[7:13]                           # right_arm 6
    out[21] = p[ARM_RIGHT_QPOS][6]                 # right_arm 마지막 관절 (고정)
    out[22] = np.clip(d[13] / DM_GRIP_MAX, 0.0, 1.0) * 2.0 - 1.0
    return out


# ---------------------------------------------------------------------------
# DM0.5 HTTP 클라이언트
# ---------------------------------------------------------------------------
class DM05Client:
    def __init__(self, url: str, robot_type: str, control_mode: str,
                 speed: str, jpeg_quality: int, timeout: float) -> None:
        self.url = url
        self.robot_type = robot_type
        self.control_mode = control_mode
        self.speed = speed
        self.jpeg_quality = jpeg_quality
        self.timeout = timeout

    def _post(self, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"DM0.5 HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"DM0.5 연결 실패 ({self.url}): {e.reason}") from e
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"DM0.5 응답 JSON 파싱 실패: {raw[:200]!r}") from e

    def infer(self, prompt: str, state14: np.ndarray, images: dict) -> tuple[np.ndarray, dict]:
        """(N,14) float32 액션 청크와 metadata를 반환. 실패는 예외로 올린다."""
        payload = {
            "observation": {
                "prompt": prompt,
                "robot_type": self.robot_type,
                "control_mode": self.control_mode,
                "speed": self.speed,
                "state": [float(x) for x in state14],
                "images": images,
            }
        }
        resp = self._post(payload)
        if not isinstance(resp, dict) or "actions" not in resp:
            raise RuntimeError(f"DM0.5 응답에 actions 없음: {list(resp)[:8]}")
        actions = np.asarray(resp["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != DM_ACTION_DIM:
            raise RuntimeError(f"DM0.5 actions 모양 이상: {actions.shape} (기대 (N,{DM_ACTION_DIM}))")
        if actions.shape[0] < 1:
            raise RuntimeError("DM0.5 actions가 비어 있음")
        if not np.all(np.isfinite(actions)):
            raise RuntimeError("DM0.5 actions에 비유한값(NaN/Inf) 포함")
        meta = resp.get("metadata", {})
        if not isinstance(meta, dict):
            meta = {"raw": meta}
        return actions, meta


# ---------------------------------------------------------------------------
# 연결별 상태 (청크 큐 + 위치)
# ---------------------------------------------------------------------------
class DM05Session:
    """연결 하나의 청크 큐. 청크 50개 중 앞 exec_steps 개만 실행한다."""

    def __init__(self, client: DM05Client, exec_steps: int, default_prompt: str,
                 trace_path: str | None = None) -> None:
        self.client = client
        self.exec_steps = exec_steps
        self.default_prompt = default_prompt
        self.trace_path = trace_path
        self.queue: np.ndarray | None = None   # (exec_steps, 23) float32
        self.pos = 0
        self.chunk_idx = 0

    def reset(self) -> None:
        self.queue = None
        self.pos = 0

    def _trace(self, record: dict) -> None:
        if not self.trace_path:
            return
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    def _fetch_chunk(self, obs: dict) -> None:
        # 카메라 (없으면 ValueError)
        images: dict = {}
        for suffix, slot in CAMERA_SUFFIXES.items():
            arr = _find_camera(obs, suffix)
            if arr is None:
                raise ValueError(f"새 청크가 필요한데 카메라 관측이 없음: {suffix}")
            images[slot] = _encode_jpeg(arr, self.client.jpeg_quality)

        proprio = obs.get("robot_r1::proprio")
        if proprio is None:
            raise ValueError("새 청크가 필요한데 proprio 관측이 없음")
        proprio = np.asarray(proprio, dtype=np.float32).reshape(-1)
        state14 = build_state14(proprio)

        subtask = obs.get("subtask")
        prompt = subtask if isinstance(subtask, str) and subtask.strip() else self.default_prompt

        t0 = time.monotonic()
        actions, meta = self.client.infer(prompt, state14, images)
        latency_ms = (time.monotonic() - t0) * 1000.0

        if actions.shape[0] < self.exec_steps:
            raise RuntimeError(
                f"DM0.5 청크가 exec_steps보다 짧음: {actions.shape[0]} < {self.exec_steps}")

        # 청크 전체를 23차원으로 변환해 두고 앞 exec_steps 개만 실행
        converted = np.stack([dm14_to_r1pro23(a, proprio) for a in actions], axis=0)
        self.queue = np.ascontiguousarray(converted[: self.exec_steps], dtype=np.float32)
        self.pos = 0
        self.chunk_idx += 1

        # trace: 실행 구간에서 dm 관절(그리퍼 제외 12개)의 max |action - state14|
        seg = actions[: self.exec_steps]
        joint_idx = list(range(0, 6)) + list(range(7, 13))
        max_dj = float(np.max(np.abs(seg[:, joint_idx] - state14[joint_idx]))) if seg.size else 0.0
        self._trace({
            "t": time.time(),
            "chunk": self.chunk_idx,
            "prompt": prompt,
            "state14": [float(x) for x in state14],
            "action0": [float(x) for x in actions[0]],
            "max_abs_delta_joint": max_dj,
            "gripper_left_cmd": float(self.queue[0, 14]),
            "gripper_right_cmd": float(self.queue[0, 22]),
            "latency_ms": latency_ms,
            "metadata": meta,
        })
        logger.info(
            "chunk %d: latency=%.0f ms, max|Δjoint|=%.4f, n_dm=%d, exec=%d",
            self.chunk_idx, latency_ms, max_dj, actions.shape[0], self.exec_steps)

    def act(self, obs: dict) -> np.ndarray:
        if self.queue is None or self.pos >= self.exec_steps:
            self._fetch_chunk(obs)
        action = self.queue[self.pos]
        self.pos += 1
        return np.ascontiguousarray(action, dtype=np.float32)


# ---------------------------------------------------------------------------
# websocket 서버 (network_utils.WebsocketPolicyServer에서 벤더링)
# ---------------------------------------------------------------------------
class WebsocketPolicyServer:
    def __init__(self, client: DM05Client, host="0.0.0.0", port=8040,
                 exec_steps: int = 25, default_prompt: str = "",
                 trace_path: str | None = None, metadata=None):
        self._client = client
        self._host, self._port = host, port
        self._exec_steps = exec_steps
        self._default_prompt = default_prompt
        self._trace_path = trace_path
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
        session = DM05Session(self._client, self._exec_steps,
                              self._default_prompt, self._trace_path)
        packer = Packer()
        await websocket.send(packer.pack(self._metadata))
        while True:
            try:
                msg = unpackb(await websocket.recv())
                if "reset" in msg:
                    session.reset()
                    continue
                t0 = time.monotonic()
                action = await asyncio.to_thread(session.act, msg)
                await websocket.send(packer.pack({
                    "action": action,
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
    proprio = np.zeros(STATE_DIM, dtype=np.float32)
    proprio[ARM_LEFT_QPOS] = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.0]
    proprio[ARM_RIGHT_QPOS] = [-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, 0.0]
    proprio[GRIPPER_LEFT_QPOS] = 0.02
    proprio[GRIPPER_RIGHT_QPOS] = 0.02
    proprio[TRUNK_QPOS] = [0.0, 0.0, 0.0, 0.0]
    gray = np.full((128, 128, 4), 128, dtype=np.uint8)
    return {
        "robot_r1::robot_r1:zed_link:Camera:0::rgb": gray,
        "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": gray,
        "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": gray,
        "robot_r1::proprio": proprio,
        "task_id": np.array([93], dtype=np.int64),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8040)
    parser.add_argument("--dm05-url", default="http://127.0.0.1:7891/v1/infer")
    parser.add_argument("--exec-steps", type=int, default=25,
                        help="청크 50개 중 실행할 앞쪽 개수 (receding horizon)")
    parser.add_argument("--robot-type", default="DOS W1")
    parser.add_argument("--control-mode", default="joint")
    parser.add_argument("--speed", default="0.5", help="OpenDM 기본값과 같은 문자열")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--http-timeout", type=float, default=60.0)
    parser.add_argument("--trace", default=None, help="청크마다 jsonl 한 줄 기록")
    parser.add_argument("--self-test", action="store_true",
                        help="더미 관측 2스텝만 돌리고 종료 (실제 DM0.5 서비스 필요)")
    args = parser.parse_args()

    assert 1 <= args.exec_steps <= DM_CHUNK, f"exec_steps 범위: 1..{DM_CHUNK}"
    if args.trace:
        Path(args.trace).parent.mkdir(parents=True, exist_ok=True)

    client = DM05Client(
        url=args.dm05_url,
        robot_type=args.robot_type,
        control_mode=args.control_mode,
        speed=args.speed,
        jpeg_quality=args.jpeg_quality,
        timeout=args.http_timeout,
    )

    if args.self_test:
        print(f"[self-test] DM0.5: {args.dm05_url}")
        session = DM05Session(client, args.exec_steps, args.prompt, args.trace)
        session.reset()
        obs = make_dummy_obs()
        for i in range(2):
            t0 = time.monotonic()
            action = session.act(obs)
            dt = (time.monotonic() - t0) * 1000.0
            assert action.shape == (ACTION_DIM,), action.shape
            assert action.dtype == np.float32, action.dtype
            assert np.all(np.isfinite(action)), "비유한값 포함"
            print(f"  step {i}: shape={tuple(action.shape)} dtype={action.dtype} "
                  f"range=[{action.min():.3f}, {action.max():.3f}] ({dt:.0f} ms)")
        assert session.chunk_idx == 1, (
            f"두 번째 호출이 HTTP를 다시 불렀음 (chunk_idx={session.chunk_idx})")
        print(f"[self-test] ✅ 통과 (chunk_idx={session.chunk_idx}, pos={session.pos})")
        return

    metadata = {
        "policy": "dm05_b1k_adapter",
        "exec_steps": args.exec_steps,
        "hl": None,
        "wm": None,
        "robot_type": args.robot_type,
        "dm05_url": args.dm05_url,
    }
    server = WebsocketPolicyServer(
        client, args.host, args.port,
        exec_steps=args.exec_steps,
        default_prompt=args.prompt,
        trace_path=args.trace,
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
