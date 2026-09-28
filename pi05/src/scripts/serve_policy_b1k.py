"""학습한 체크포인트를 웹소켓 정책 서버로 띄운다 (OmniGibson 평가용).

    third_party/openpi/.venv/bin/python scripts/serve_policy_b1k.py [--port 8000] [--device cpu]

## 왜 서버로 띄우나

시뮬레이터(OmniGibson)와 정책(openpi/JAX)은 **파이썬 환경이 다르다.**
OmniGibson 은 conda `behavior`, 정책은 `third_party/openpi/.venv` 다. 한 프로세스에
못 넣으므로 웹소켓으로 잇는다. 공식 baseline 도 같은 구조이고, BEHAVIOR 쪽
`network_utils.py` 는 아예 "Adapted from openpi" 라고 밝히고 있어 프로토콜이 같다.

## 이 서버가 하는 변환

시뮬레이터가 주는 것과 정책이 받는 것의 이름·형식을 맞춘다. **값 자체는
바꿀 것이 없다** — 오늘 대조해 보니 관절 상태 61차원의 배열 순서와 액션
23차원의 구성이 우리 학습 데이터와 시뮬레이터가 완전히 같았다:

    시뮬 base_qvel 0:3 / arm_left 3:10 / gripper_left 24:26 /
         arm_right 28:35 / gripper_right 49:51 / trunk 53:57
    = configs/r1pro_policy.py 의 STATE_SELECT 가 고르는 바로 그 인덱스들

액션도 base 3 + trunk 4 + 왼팔 7 + 왼그리퍼 1 + 오른팔 7 + 오른그리퍼 1 = 23 으로
순서까지 일치하고, 시뮬레이터 컨트롤러가 `use_delta_commands: false` 라
**절대 관절값**을 받는데 우리 정책도 절대값을 낸다.

## 주고받는 것

요청: {"image", "left_wrist_image", "right_wrist_image": (224,224,3) uint8,
       "state": (61,) float32, "prompt": str}
응답: {"actions": (32,23) float32}   ← 청크를 통째로 준다

**청크를 그대로 돌려주는 이유**는 실행 쪽(receding horizon: 32 중 16 실행)을
호출자가 정하게 하기 위해서다. 이미지 축소(1080→224)도 호출자 쪽에서 한다 —
그래야 웹소켓으로 오가는 양이 23분의 1이 된다.
"""
import argparse, os, pathlib, sys, time

ROOT = pathlib.Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--host", default="0.0.0.0")
ap.add_argument("--device", choices=("cpu", "gpu"), default="cpu",
                help="시뮬레이터가 GPU 를 쓰므로 기본은 cpu. 8GB 카드에 둘 다는 안 들어간다")
ap.add_argument("--config", default=os.environ.get("COOKING_CONFIG", "pi05_cooking_top20"))
ap.add_argument("--ckpt", default=os.environ.get(
    "COOKING_CKPT", "checkpoints/pi05_cooking_top20/run2/12999"))
a = ap.parse_args()

os.environ.setdefault("HF_LEROBOT_HOME", str(ROOT / "data/lerobot"))
os.environ["HF_HUB_OFFLINE"] = "1"
if a.device == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
sys.path.insert(0, str(ROOT / "configs"))

import numpy as np
import lerobot_patch, cooking_config, r1pro_policy, policy_loader

RAW_STATE_DIM = r1pro_policy.RAW_STATE_DIM   # 61


class B1KPolicy:
    """시뮬레이터의 관측을 정책 입력 이름으로 바꿔 주는 얇은 껍데기."""

    def __init__(self, policy):
        self._policy = policy
        self.n = 0

    def infer(self, obs: dict) -> dict:
        state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
        if state.shape[0] != RAW_STATE_DIM:
            # 조용히 틀린 값을 내는 것이 이 프로젝트에서 가장 비쌌던 실수라
            # 차원이 다르면 즉시 막는다.
            raise ValueError(
                f"state 차원이 {state.shape[0]} 이다 — {RAW_STATE_DIM} 이어야 한다. "
                "시뮬레이터의 proprio_obs 구성이 eval/r1pro.yaml 과 같은지 확인할 것")
        t0 = time.time()
        out = self._policy.infer({
            "observation/image": np.asarray(obs["image"]),
            "observation/left_wrist_image": np.asarray(obs["left_wrist_image"]),
            "observation/right_wrist_image": np.asarray(obs["right_wrist_image"]),
            "observation/state": state,
            "prompt": obs["prompt"],
        })
        self.n += 1
        acts = np.asarray(out["actions"], dtype=np.float32)
        print(f"  [{self.n:4d}] {time.time()-t0:5.2f}초  '{obs['prompt'][:52]}'  "
              f"-> {acts.shape}", flush=True)
        return {"actions": acts}

    def reset(self) -> None:
        pass


def main():
    lerobot_patch.apply()
    # task_subset.apply() 는 부르지 않는다 — 여기서는 데이터셋을 만들지 않는다.
    cooking_config.register()
    import jax
    from openpi.training import config as _config
    from openpi.serving import websocket_policy_server

    cfg = _config.get_config(a.config)
    print(f"백엔드 {jax.devices()}\n설정 {a.config}\n체크포인트 {ROOT / a.ckpt}")
    t0 = time.time()
    policy, _ = policy_loader.load_policy(cfg, ROOT / a.ckpt)
    print(f"정책 로드 {time.time()-t0:.0f}초")

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=B1KPolicy(policy), host=a.host, port=a.port,
        metadata={"config": a.config, "ckpt": str(a.ckpt),
                  "action_horizon": int(cfg.model.action_horizon)})
    print(f"\n대기 중 — ws://{a.host}:{a.port}  (Ctrl+C 로 종료)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
