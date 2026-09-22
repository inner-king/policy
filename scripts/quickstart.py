"""OmniGibson 최소 예제: 씬 + R1Pro 로봇을 헤드리스로 띄워 랜덤 행동 후 카메라 이미지를 저장합니다.

    python -u scripts/quickstart.py                       # Rs_int, 100 step, ~/og_outputs
    python -u scripts/quickstart.py --scene Beechwood_0_int --steps 300 --out /tmp/og
    python -u scripts/quickstart.py --no-viewer           # 1280x720 뷰어 카메라 렌더링 끄기 (더 빠름)
"""
import argparse
import os

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")  # omnigibson import 전에 설정해야 함

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="Rs_int", help="datasets/behavior-1k-assets/scenes 안의 씬 이름, 또는 empty")
ap.add_argument("--robot", default="r1pro", help="소문자 모델 ID (omnigibson-robot-assets/models 폴더 이름)")
ap.add_argument("--steps", type=int, default=100)
ap.add_argument("--cam", type=int, default=128, help="로봇 카메라 해상도 (H=W)")
ap.add_argument("--rooms", nargs="*", default=None, help="일부 방만 로드: 예) --rooms kitchen living_room")
ap.add_argument("--out", default=os.path.expanduser("~/og_outputs"))
ap.add_argument("--no-viewer", action="store_true", help="gm.RENDER_VIEWER_CAMERA=False")
args = ap.parse_args()

import torch as th  # noqa: E402
import omnigibson as og  # noqa: E402
from omnigibson.macros import gm  # noqa: E402

if args.no_viewer:
    gm.RENDER_VIEWER_CAMERA = False  # og.Environment 생성 전에 설정

if args.scene == "empty":
    scene = {"type": "Scene"}
else:
    scene = {"type": "InteractiveTraversableScene", "scene_model": args.scene, "load_room_types": args.rooms}

cfg = {
    "scene": scene,
    "robots": [{
        "model": args.robot,
        "name": "robot0",  # 이름을 고정해야 obs 키와 저장 파일 이름이 매번 같음
        "obs_modalities": ["rgb", "depth"],
        "action_normalize": True,
        "action_type": "continuous",
        "sensor_config": {"VisionSensor": {"sensor_kwargs": {"image_height": args.cam, "image_width": args.cam}}},
    }],
    "task": {"type": "DummyTask"},
}

env = og.Environment(configs=cfg)
obs, info = env.reset()
robot = env.robots[0]

# obs 구조: obs[로봇 이름][센서 이름][modality]
for k, v in obs[robot.name].items():
    print(k, {m: tuple(x.shape) for m, x in v.items()} if isinstance(v, dict) else tuple(v.shape), flush=True)
print("action 순서:", robot.controller_order, "차원:", robot.action_dim, flush=True)

for i in range(args.steps):
    action = th.as_tensor(robot.action_space.sample(), dtype=th.float32) * 0.1
    obs, reward, terminated, truncated, info = env.step(action)
    if i % 50 == 0:
        print("step", i, flush=True)

from PIL import Image  # noqa: E402

os.makedirs(args.out, exist_ok=True)
for sensor_name, sensor_obs in obs[robot.name].items():
    if isinstance(sensor_obs, dict) and "rgb" in sensor_obs:
        img = sensor_obs["rgb"][..., :3].cpu().numpy()
        path = os.path.join(args.out, sensor_name.replace(":", "_") + ".png")
        Image.fromarray(img).save(path)
        print("saved", path, flush=True)

og.shutdown()  # 프로세스가 여기서 종료됨. 이후 코드는 실행되지 않음
