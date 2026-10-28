"""RemoteVectorEnv against a fake VectorEnv in a separate interpreter.

ROBOTWIN_TEST_SIM_PYTHON selects the server interpreter (default: this one).
Point it at a different venv to exercise a real cross-version boundary.
"""

import importlib
import os
import pickle
import sys
import time

import numpy as np
import pytest

from robotwin.envs import remote_vector_env
from robotwin.envs.remote_vector_env import RemoteVectorEnv

SIM_PYTHON = os.environ.get("ROBOTWIN_TEST_SIM_PYTHON", sys.executable)
FAKE_TARGET = "robotwin.tests.fake_vector_env:FakeVectorEnv"


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ROBOTWIN_SIM_PYTHON", SIM_PYTHON)
    venv = RemoteVectorEnv({"task_name": "fake"}, 2, env_seeds=np.array([5, 6]), target=FAKE_TARGET)
    yield venv
    venv.shutdown()


def test_forwards_vector_env_calls(env):
    env.reset()
    obs = env.get_obs()
    assert [o["seed"] for o in obs] == [5, 6]
    assert obs[1]["full_image"].dtype == np.uint8

    obs, reward, terminated, truncated, info = env.step(np.ones((2, 3, 14), np.float32))
    assert reward == [42.0, 42.0]
    assert [o["step"] for o in obs] == [3, 3]
    assert terminated == truncated == [False, False]

    obs_list, *_ = env.step_with_full_obs(np.zeros((2, 4, 14)))
    assert len(obs_list) == 4 and len(obs_list[0]) == 2

    assert env.check_seeds(np.array([7, 8])) == [{"seed": 7}, {"seed": 8}]


def test_array_indices_arrive_as_lists(env):
    env.reset(env_idx=np.array([1]), env_seeds=np.array([9]))
    assert [o["seed"] for o in env.get_obs()] == [5, 9]


def test_clearing_close_restarts_server(env):
    env.reset(env_seeds=[7, 8])
    env.step(np.ones((2, 1, 14)))
    old_proc = env._proc
    env.close()
    assert old_proc.wait(timeout=10) == 0
    obs = env.get_obs()
    assert env._proc is not old_proc
    assert [o["seed"] for o in obs] == [5, 6]
    assert [o["step"] for o in obs] == [0, 0]


def test_non_clearing_close_keeps_server(env):
    env.reset(env_seeds=[7, 8])
    env.step(np.ones((2, 1, 14)))
    proc = env._proc
    env.close(clear_cache=False)
    obs = env.get_obs()
    assert env._proc is proc
    assert [o["seed"] for o in obs] == [7, 8]
    assert [o["step"] for o in obs] == [0, 0]


def test_remote_exception_carries_traceback(env):
    with pytest.raises(RuntimeError, match="fake failure") as exc_info:
        env._call("fail")
    assert "Traceback" in str(exc_info.value)
    assert env.get_obs()


def test_private_methods_are_rejected(env):
    with pytest.raises(RuntimeError, match="not a public VectorEnv method"):
        env._call("_secret")


def test_server_crash_reports_exit_code(env):
    with pytest.raises(RuntimeError, match="exit code 3"):
        env._call("crash")


def test_shutdown_stops_server(env):
    proc = env._proc
    env.shutdown()
    assert proc.poll() == 0
    with pytest.raises(RuntimeError, match="shut down"):
        env.get_obs()


def test_server_restarts_after_configured_full_resets(monkeypatch):
    monkeypatch.setenv("ROBOTWIN_SIM_PYTHON", SIM_PYTHON)
    task_config = {"task_name": "fake", "sim_restart_resets": 2}
    venv = RemoteVectorEnv(task_config, 2, env_seeds=[5, 6], target=FAKE_TARGET)
    try:
        assert "sim_restart_resets" in task_config
        first = venv._proc
        venv.reset(env_seeds=[7, 8])
        venv.reset(env_idx=[0], env_seeds=[9])
        venv.reset(env_seeds=[7, 8])
        assert venv._proc is first
        venv.reset(env_seeds=[3, 4])
        assert first.wait(timeout=10) == 0
        assert venv._proc is not first
        assert [o["seed"] for o in venv.get_obs()] == [3, 4]
    finally:
        venv.shutdown()


def test_failed_respawn_stops_server_and_retries(env):
    init_args = env._init_args
    env.close()
    env._init_args = ({}, 2)
    with pytest.raises(RuntimeError, match="KeyError"):
        env.get_obs()
    assert env._proc.wait(timeout=10) == 0
    env._init_args = init_args
    assert [o["seed"] for o in env.get_obs()] == [5, 6]


def test_close_after_shutdown_does_not_restart(env):
    env.shutdown()
    env.close()
    with pytest.raises(RuntimeError, match="shut down"):
        env.get_obs()


def test_torch_actions_are_sent_as_numpy(env):
    torch = pytest.importorskip("torch")
    _, reward, *_ = env.step(torch.ones(2, 3, 14))
    assert reward == [42.0, 42.0]


def test_unpicklable_argument_leaves_channel_usable(env):
    with pytest.raises((pickle.PicklingError, AttributeError)):
        env._call("check_seeds", lambda: None)
    assert env.check_seeds([1]) == [{"seed": 1}]


def test_corrupt_request_gets_error_reply(env):
    env._conn.send_bytes(b"not a pickle")
    assert remote_vector_env.recv(env._conn)[0] == "error"
    assert env.get_obs()


def test_interrupted_call_marks_env_unusable(env, monkeypatch):
    def interrupted(conn):
        raise KeyboardInterrupt

    monkeypatch.setattr(remote_vector_env, "recv", interrupted)
    with pytest.raises(KeyboardInterrupt):
        env.get_obs()
    monkeypatch.undo()
    with pytest.raises(RuntimeError, match="unusable"):
        env.get_obs()


def test_shutdown_of_wedged_server_is_bounded(env, monkeypatch):
    monkeypatch.setattr(remote_vector_env, "SHUTDOWN_TIMEOUT_S", 1)
    env._conn.send_bytes(pickle.dumps(("slow", (5,), {})))
    start = time.monotonic()
    env.shutdown()
    assert time.monotonic() - start < 3
    assert env._proc.wait(timeout=10) == 0


def test_dispatcher_selects_remote_when_sim_python_set(monkeypatch):
    import robotwin.envs

    monkeypatch.setenv("ROBOTWIN_SIM_PYTHON", SIM_PYTHON)
    monkeypatch.delitem(sys.modules, "robotwin.envs.vector_env", raising=False)
    monkeypatch.delattr(robotwin.envs, "vector_env", raising=False)
    vector_env = importlib.import_module("robotwin.envs.vector_env")
    assert vector_env.VectorEnv is RemoteVectorEnv
