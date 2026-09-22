"""BEHAVIOR 태스크 예제: configs/r1pro_behavior.yaml을 불러와 미리 샘플링된 활동을 실행합니다.

    python -u scripts/behavior_task_demo.py                                   # picking_up_trash
    python -u scripts/behavior_task_demo.py --activity make_microwave_popcorn
    python -u scripts/behavior_task_demo.py --list                            # 이 씬에서 가능한 활동 목록

활동은 미리 샘플링된 인스턴스가 있어야 합니다:
    datasets/2026-challenge-task-instances/scenes/<씬>/json/<씬>_task_<활동>_0_0_template.json
"""
import argparse
import glob
import os

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")

ap = argparse.ArgumentParser()
ap.add_argument("--activity", default="picking_up_trash")
ap.add_argument("--scene", default=None, help="기본: yaml의 house_double_floor_lower")
ap.add_argument("--all-rooms", action="store_true", help="yaml의 load_room_types(living_room, kitchen) 제한 해제")
ap.add_argument("--steps", type=int, default=100)
ap.add_argument("--list", action="store_true")
args = ap.parse_args()

import yaml  # noqa: E402
import omnigibson as og  # noqa: E402
from omnigibson.macros import gm  # noqa: E402

cfg = yaml.safe_load(open(os.path.join(og.example_config_path, "r1pro_behavior.yaml")))
scene = args.scene or cfg["scene"]["scene_model"]

if args.list:
    pat = os.path.join(gm.DATA_PATH, "2026-challenge-task-instances", "scenes", scene, "json", f"{scene}_task_*_0_0_template.json")
    names = sorted(os.path.basename(p)[len(f"{scene}_task_"):-len("_0_0_template.json")] for p in glob.glob(pat))
    print(f"{scene}: {len(names)}개 활동")
    print("\n".join(names))
    raise SystemExit(0)

cfg["scene"]["scene_model"] = scene
if args.all_rooms:
    cfg["scene"]["load_room_types"] = None
cfg["task"]["activity_name"] = args.activity

env = og.Environment(configs=cfg)
env.reset()
print("activity:", env.task.activity_name, "scene:", scene, flush=True)
robot = env.robots[0]
for i in range(args.steps):
    obs, reward, terminated, truncated, info = env.step(robot.action_space.sample() * 0.1)
    if terminated or truncated:
        print(f"episode finished at step {i + 1}", flush=True)
        env.reset()
print("done", flush=True)
og.shutdown()
