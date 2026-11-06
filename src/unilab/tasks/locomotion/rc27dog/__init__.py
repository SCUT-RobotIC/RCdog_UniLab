"""Hydra-owned Manager-Based registrations for the 27dog wheel-legged robot."""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env

registry.register_env_config("RC27DogWheelFlat", ManagerBasedRlEnvCfg)
registry.register_env("RC27DogWheelFlat", make_manager_based_rl_env, sim_backend="mujoco")
registry.register_env("RC27DogWheelFlat", make_manager_based_rl_env, sim_backend="mjwarp")

__all__: list[str] = []
