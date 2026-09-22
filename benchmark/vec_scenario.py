"""og.VectorEnvironment scaling benchmark (modelled on omnigibson/examples/environments/vector_env_demo.py).

usage: python vec_scenario.py --num-envs N [--steps 200] [--warmup 10]
Prints @@MARK / @@STAT lines for monitor.py.
"""
import argparse, json, os, time

os.environ["OMNIGIBSON_HEADLESS"] = "1"
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


def mark(name):
    print(f"@@MARK {name}", flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--num-envs", type=int, required=True)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--warmup", type=int, default=10)
a = ap.parse_args()

t_start = time.time()
import yaml  # noqa: E402
import omnigibson as og  # noqa: E402
from omnigibson.macros import gm  # noqa: E402

gm.HEADLESS = True
gm.RENDER_VIEWER_CAMERA = False
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = False
gm.ENABLE_OBJECT_STATES = False
mark("import_done")

config_filename = os.path.join(og.example_config_path, "franka_vector_env.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
config["scene"]["load_object_categories"] = ["floors", "walls", "coffee_table"]
print("CFG", json.dumps({"num_envs": a.num_envs, "scene": config["scene"]["scene_model"],
                         "load_object_categories": config["scene"]["load_object_categories"],
                         "robot": config["robots"][0]["type"],
                         "cam": config["robots"][0]["sensor_config"]["VisionSensor"]["sensor_kwargs"]}), flush=True)

vec_env = og.VectorEnvironment(a.num_envs, config)
mark("env_created")

n_objs = sum(len(e.scene.objects) for e in vec_env.envs)
print(f"@@STAT {json.dumps({'n_objects': n_objs, 'num_envs': a.num_envs})}", flush=True)


def one_step():
    actions = [e.action_space.sample() for e in vec_env.envs]
    vec_env.step(actions)


for _ in range(a.warmup):
    one_step()
mark("warmup_done")
t0 = time.time()
for _ in range(a.steps):
    one_step()
dt = time.time() - t0
mark("steps_done")
sps = a.steps / dt
print(f"@@STAT {json.dumps({'num_envs': a.num_envs, 'steps': a.steps, 'step_sec': round(dt, 2), 'vec_steps_per_sec': round(sps, 2), 'total_env_steps_per_sec': round(a.num_envs * sps, 2), 'per_env_steps_per_sec': round(sps, 2), 'sim_realtime_factor': round(a.steps / 30 / dt, 3), 'startup_to_ready_sec': round(t0 - t_start, 1)})}", flush=True)
og.shutdown()
