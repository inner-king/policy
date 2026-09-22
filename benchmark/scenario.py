"""OmniGibson resource-benchmark scenarios. Prints @@MARK / @@STAT lines for monitor.py.

usage: python scenario.py --scene {empty|<scene_model>} [--rooms a b ...] [--robot R1Pro] \
         [--cam 128] [--modalities rgb depth] [--task behavior:<activity>] [--steps 300] [--gpu-dynamics]
"""
import argparse, json, os, sys, time

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


def mark(name):
    print(f"@@MARK {name}", flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="empty")
ap.add_argument("--rooms", nargs="*", default=None, help="load_room_types")
ap.add_argument("--robot", default=None)
ap.add_argument("--cam", type=int, default=128, help="robot camera H=W")
ap.add_argument("--modalities", nargs="*", default=["rgb", "depth"])
ap.add_argument("--task", default=None, help="behavior:<activity_name>")
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--gpu-dynamics", action="store_true")
ap.add_argument("--object-states", action="store_true")
a = ap.parse_args()

t_start = time.time()
import torch as th  # noqa: E402
import omnigibson as og  # noqa: E402
from omnigibson.macros import gm  # noqa: E402

gm.USE_GPU_DYNAMICS = a.gpu_dynamics
gm.ENABLE_OBJECT_STATES = a.object_states or (a.task is not None)
gm.ENABLE_TRANSITION_RULES = a.task is not None
mark("import_done")

if a.scene == "empty":
    scene = {"type": "Scene"}
else:
    scene = {"type": "InteractiveTraversableScene", "scene_model": a.scene, "load_room_types": a.rooms}

robots = []
if a.robot:
    robots.append({
        "model": a.robot.lower(),
        "obs_modalities": list(a.modalities) + ["proprio"] if a.modalities else ["proprio"],
        "action_normalize": True,
        "action_type": "continuous",
        "sensor_config": {"VisionSensor": {"sensor_kwargs": {"image_height": a.cam, "image_width": a.cam}}},
    })

if a.task and a.task.startswith("behavior:"):
    task = {"type": "BehaviorTask", "activity_name": a.task.split(":", 1)[1], "activity_definition_id": 0,
            "activity_instance_id": 0, "online_object_sampling": False, "use_presampled_robot_pose": True}
    scene["include_robots"] = True
else:
    task = {"type": "DummyTask"}

cfg = {"env": {"action_frequency": 30, "rendering_frequency": 30, "physics_frequency": 120},
       "scene": scene, "robots": robots, "objects": [], "task": task}
print("CFG", json.dumps(cfg), flush=True)

env = og.Environment(configs=cfg)
mark("env_created")
env.reset()
mark("reset_done")

n_objs = len(env.scene.objects)
print(f"@@STAT {json.dumps({'n_objects': n_objs})}", flush=True)

# warm-up, then timed loop
def one_step():
    if env.robots:
        action = env.robots[0].action_space.sample() * 0.1
        env.step(th.as_tensor(action, dtype=th.float32) if not isinstance(action, th.Tensor) else action)
    else:
        env.step(th.zeros(0))

for _ in range(20):
    one_step()
mark("warmup_done")
t0 = time.time()
for _ in range(a.steps):
    one_step()
dt = time.time() - t0
mark("steps_done")
print(f"@@STAT {json.dumps({'steps': a.steps, 'step_sec': round(dt, 2), 'env_steps_per_sec': round(a.steps / dt, 2), 'sim_realtime_factor': round(a.steps / 30 / dt, 3), 'startup_to_ready_sec': round(t0 - t_start, 1)})}", flush=True)
og.shutdown()
