import os

if os.environ.get("ROBOTWIN_SIM_PYTHON"):
    from robotwin.envs.remote_vector_env import RemoteVectorEnv as VectorEnv
else:
    from robotwin.envs._local_vector_env import VectorEnv

__all__ = ["VectorEnv"]
