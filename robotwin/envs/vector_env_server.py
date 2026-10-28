"""Sim-side loop for ``RemoteVectorEnv``. Started by the client, not by users.

Serves one VectorEnv over an inherited socket and exits on EOF, so it never
outlives the env worker that started it.
"""

import argparse
import importlib
import traceback
from multiprocessing.connection import Connection

from robotwin.envs.remote_vector_env import recv, send


def _load(target):
    module_name, class_name = target.split(":")
    return getattr(importlib.import_module(module_name), class_name)


def _handle(conn, target, env):
    """Serve one request. Returns the (possibly new) env, or raises EOFError to stop."""
    try:
        name, args, kwargs = recv(conn)
    except EOFError:
        raise
    except Exception:
        send(conn, ("error", traceback.format_exc()))
        return env
    if name == "_shutdown":
        send(conn, ("ok", None))
        raise EOFError
    try:
        if name == "__init__":
            env = _load(target)(*args, **kwargs)
            result = None
        elif name.startswith("_"):
            raise AttributeError(f"{name!r} is not a public VectorEnv method")
        else:
            result = getattr(env, name)(*args, **kwargs)
        send(conn, ("ok", result))
    except (BrokenPipeError, ConnectionResetError):
        raise
    except Exception:
        send(conn, ("error", traceback.format_exc()))
    return env


def serve(conn, target):
    env = None
    try:
        while True:
            env = _handle(conn, target, env)
    except (EOFError, BrokenPipeError, ConnectionResetError):
        pass  # client finished or is already gone
    finally:
        if env is not None:
            env.close(clear_cache=True)
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    serve(Connection(args.fd), args.target)


if __name__ == "__main__":
    main()
