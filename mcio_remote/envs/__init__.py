from gymnasium.envs.registration import register

from . import base_env, mcio_env, minerl_env

__all__ = [
    "base_env",
    "mcio_env",
    "minerl_env",
]

register(
    id="mcio_env/MCioEnv-v0",
    entry_point="mcio_remote.envs:MCioEnv",
)
