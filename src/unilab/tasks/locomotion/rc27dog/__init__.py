"""Hydra-owned Manager-Based registrations for the 27dog wheel-legged robot."""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env

for _task_name in (
    "RC27DogWheelFlat",
    "RC27DogWheelRough",
    "RC27DogWheelRoughStones",
    "RC27DogWheelLowBar",
    "RC27DogWheelLowWall",
    "RC27DogWheelBrokenBridge",
):
    registry.register_env_config(_task_name, ManagerBasedRlEnvCfg)
    registry.register_env(_task_name, make_manager_based_rl_env, sim_backend="mujoco")
    registry.register_env(_task_name, make_manager_based_rl_env, sim_backend="mjwarp")

__all__: list[str] = []
