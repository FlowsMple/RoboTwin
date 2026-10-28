"""Check that RemoteVectorEnv returns exactly what the in-process VectorEnv returns.

Needs a GPU, RoboTwin assets, and two interpreters:

  SIM=<sim venv python> POLICY=<policy venv python>
  $SIM    robotwin/tests/parity_adjust_bottle.py dump local.pkl
  ROBOTWIN_SIM_PYTHON=$SIM $POLICY robotwin/tests/parity_adjust_bottle.py dump remote.pkl
  $POLICY robotwin/tests/parity_adjust_bottle.py compare local.pkl remote.pkl

ROBOTWIN_ASSETS_PATH defaults to the RoboTwin root, where ./assets/embodiments lives.
"""

import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

ROBOTWIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROBOTWIN_ROOT))

TASK_CONFIG = {
    "task_name": "adjust_bottle",
    "step_lim": 200,
    "planner_backend": "mplib",
    "render_freq": 0,
    "episode_num": 1,
    "use_seed": False,
    "save_freq": 15,
    "embodiment": ["aloha-agilex"],
    "language_num": 1,
    "domain_randomization": {
        "random_background": False,
        "cluttered_table": False,
        "clean_background_rate": 1,
        "random_head_camera_dis": 0,
        "random_table_height": 0,
        "random_light": False,
        "crazy_random_light_rate": 0,
    },
    "camera": {
        "head_camera_type": "D435",
        "wrist_camera_type": "D435",
        "collect_head_camera": True,
        "collect_wrist_camera": True,
    },
    "data_type": {
        "rgb": True, "third_view": False, "depth": False, "pointcloud": False,
        "observer": False, "endpose": False, "qpos": True,
        "mesh_segmentation": False, "actor_segmentation": False,
    },
    "pcd_down_sample_num": 1024,
    "pcd_crop": True,
    "clear_cache_freq": 8,
    "collect_data": False,
    "eval_video_log": False,
}


def dump(out_path):
    os.environ["ASSETS_PATH"] = os.environ.get("ROBOTWIN_ASSETS_PATH", str(ROBOTWIN_ROOT))
    from robotwin.envs.vector_env import VectorEnv

    env = VectorEnv({**TASK_CONFIG, "save_path": tempfile.mkdtemp()}, n_envs=2, env_seeds=[0, 1])
    env.reset()
    first_obs = env.get_obs()
    actions = np.random.RandomState(0).randn(2, 5, 14).astype(np.float32) * 0.01
    step = env.step(actions)
    full = env.step_with_full_obs(actions)
    env.close()
    getattr(env, "shutdown", lambda: None)()
    with open(out_path, "wb") as f:
        pickle.dump({"backend": type(env).__name__, "first_obs": first_obs, "step": step, "full": full}, f)
    print(f"{type(env).__name__}: reward={step[1]} terminated={step[2]}")


def diff(a, b, path=""):
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return [f"{path}: keys {sorted(a)} != {sorted(b)}"]
        return [m for k in a for m in diff(a[k], b[k], f"{path}.{k}")]
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: len {len(a)} != {len(b)}"]
        return [m for i, (x, y) in enumerate(zip(a, b)) for m in diff(x, y, f"{path}[{i}]")]
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        same = np.shape(a) == np.shape(b) and np.asarray(a).dtype == np.asarray(b).dtype and np.array_equal(a, b)
        return [] if same else [f"{path}: arrays differ"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def compare(local_path, remote_path):
    with open(local_path, "rb") as f:
        local = pickle.load(f)
    with open(remote_path, "rb") as f:
        remote = pickle.load(f)
    assert local.pop("backend") == "VectorEnv" and remote.pop("backend") == "RemoteVectorEnv"
    mismatches = diff(local, remote)
    for line in mismatches[:50]:
        print(line)
    print("PARITY FAIL" if mismatches else "PARITY OK")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    command, *paths = sys.argv[1:]
    {"dump": dump, "compare": compare}[command](*paths)
