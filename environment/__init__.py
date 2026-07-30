from gymnasium.envs.registration import register

from .custom_env import DisasterResponseEnv

register(
    id="DisasterResponseMission-v0",
    entry_point="environment.custom_env:DisasterResponseEnv",
)

__all__ = ["DisasterResponseEnv"]
