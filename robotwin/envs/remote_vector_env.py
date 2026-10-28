"""VectorEnv proxy that runs the simulator under ``ROBOTWIN_SIM_PYTHON``.

Only the standard library is imported here, so the policy interpreter needs no
sapien, curobo, or RoboTwin ``envs`` packages. Arguments and results cross the
process boundary with plain pickle and must stay numpy/builtin; torch tensors
are converted on this side because the two interpreters may run different torch.
"""

import os
import pickle
import socket
import subprocess
import sys
import threading
import time
import weakref
from multiprocessing.connection import Connection
from pathlib import Path

ROBOTWIN_ROOT = Path(__file__).resolve().parents[2]
LOCAL_TARGET = "robotwin.envs._local_vector_env:VectorEnv"
SHUTDOWN_TIMEOUT_S = 30


def send(conn, message):
    # Plain pickle: multiprocessing's pickler shares torch tensors by fd, which a
    # Popen child cannot receive.
    conn.send_bytes(pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL))


def recv(conn):
    return pickle.loads(conn.recv_bytes())


def _as_list(value):
    return value.tolist() if hasattr(value, "tolist") else value


def _as_numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else value


def _shutdown(conn, proc):
    deadline = time.monotonic() + SHUTDOWN_TIMEOUT_S
    try:
        send(conn, ("_shutdown", (), {}))
        if conn.poll(SHUTDOWN_TIMEOUT_S):
            recv(conn)
    except (EOFError, OSError):
        pass
    conn.close()
    try:
        proc.wait(timeout=max(0.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        print(
            f"[RemoteVectorEnv] sim server pid {proc.pid} still running after "
            f"{SHUTDOWN_TIMEOUT_S}s; it exits once it reads EOF.",
            file=sys.stderr,
        )


class RemoteVectorEnv:
    """Same constructor and methods as ``VectorEnv``; the sim runs in a child process.

    The child inherits this process's environment and working directory, so GPU
    selection, ``ASSETS_PATH``, and relative paths resolve as in local mode.
    """

    def __init__(self, task_config, n_envs, env_seeds=None, instruction_type="seen", *, target=LOCAL_TARGET):
        sim_python = os.environ["ROBOTWIN_SIM_PYTHON"]
        if not os.path.isfile(sim_python):
            raise FileNotFoundError(f"ROBOTWIN_SIM_PYTHON does not exist: {sim_python}")

        child_env = dict(os.environ)
        child_env.pop("ROBOTWIN_SIM_PYTHON")
        child_env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(ROBOTWIN_ROOT), child_env.get("PYTHONPATH")) if p
        )

        task_config = dict(task_config)
        # Restart the server after this many full resets; 0 keeps it for the env's lifetime.
        self._restart_resets = max(0, int(task_config.pop("sim_restart_resets", 0) or 0))
        self._full_resets = 0

        self._command = [sim_python, "-m", "robotwin.envs.vector_env_server", "--target", target]
        self._child_env = child_env
        self._init_args = (task_config, n_envs)
        self._init_kwargs = {"env_seeds": _as_list(env_seeds), "instruction_type": instruction_type}
        self._lock = threading.Lock()
        self._proc = None
        self._finalizer = None
        self._shut = False
        self._broken = None

        self.n_envs = n_envs
        with self._lock:
            self._start()

    def _start(self):
        parent_sock, child_sock = socket.socketpair()
        with parent_sock, child_sock:
            self._proc = subprocess.Popen(
                [*self._command, "--fd", str(child_sock.fileno())],
                pass_fds=(child_sock.fileno(),),
                env=self._child_env,
            )
            self._conn = Connection(parent_sock.detach())
        self._finalizer = weakref.finalize(self, _shutdown, self._conn, self._proc)
        self._broken = None
        self._full_resets = 0
        try:
            self._request("__init__", self._init_args, self._init_kwargs)
        except BaseException:
            # A server without an env would answer every later call with an AttributeError.
            self._finalizer()
            raise

    def reset(self, env_idx=None, env_seeds=None):
        full = env_idx is None
        # The renderer's GPU memory grows across episodes. A full reset rebuilds every
        # scene anyway, so restarting the server there loses no episode state.
        if full and self._restart_resets and self._full_resets >= self._restart_resets:
            self.close()
        result = self._call("reset", env_idx=_as_list(env_idx), env_seeds=_as_list(env_seeds))
        if full:
            self._full_resets += 1
        return result

    def get_obs(self):
        return self._call("get_obs")

    def step(self, actions):
        return self._call("step", _as_numpy(actions))

    def step_with_full_obs(self, actions):
        return self._call("step_with_full_obs", _as_numpy(actions))

    def close(self, clear_cache=True):
        if not clear_cache:
            with self._lock:
                if self._finalizer is None or not self._finalizer.alive:
                    return None
            return self._call("close", clear_cache=False)
        # The renderer keeps its GPU memory until the server process exits, so a
        # clearing close stops the server; the next call starts a fresh one, just as
        # VectorEnv rebuilds its sub-envs from the constructor arguments.
        with self._lock:
            if self._finalizer is not None:
                self._finalizer()

    def check_seeds(self, seeds):
        return self._call("check_seeds", _as_list(seeds))

    def shutdown(self):
        with self._lock:
            self._shut = True
            if self._finalizer is not None:
                self._finalizer()

    def _call(self, name, *args, **kwargs):
        with self._lock:
            if self._shut:
                raise RuntimeError("RemoteVectorEnv is shut down")
            if self._broken:
                raise RuntimeError(f"RemoteVectorEnv is unusable after an interrupted {self._broken} call")
            if not self._finalizer.alive:
                self._start()
            return self._request(name, args, kwargs)

    def _request(self, name, args, kwargs):
        # Pickle before sending so an unpicklable argument fails without desyncing the channel.
        request = pickle.dumps((name, args, kwargs), protocol=pickle.HIGHEST_PROTOCOL)
        try:
            self._conn.send_bytes(request)
            status, payload = recv(self._conn)
        except (EOFError, OSError) as exc:
            self._broken = name
            try:
                code = self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                code = None
            raise RuntimeError(
                f"RoboTwin sim server died during {name} (exit code {code}); "
                "its traceback is in this worker's stderr."
            ) from exc
        except BaseException:
            # A reply may still be in flight; the next call would read it.
            self._broken = name
            raise
        if status == "error":
            raise RuntimeError(f"RoboTwin sim server failed in {name}:\n{payload}")
        return payload
