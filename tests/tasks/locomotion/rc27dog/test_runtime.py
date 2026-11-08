from __future__ import annotations

from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.base import registry
from unilab.base.config_adapter import EnvConfigAdapter
from unilab.base.registry import ensure_registries

_TASKS = (
    "rc27dog_flat",
    "rc27dog_rough",
    "rc27dog_rough_stones",
    "rc27dog_low_bar",
    "rc27dog_low_wall",
    "rc27dog_broken_bridge",
)


def test_rc27dog_tasks_reset_partial_reset_and_step() -> None:
    """Exercise the public TorchEnv lifecycle for every migrated owner."""
    ensure_registries()
    root = Path(__file__).parents[4]
    config_dir = root / "src" / "unilab" / "conf" / "flashsac"

    for task in _TASKS:
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
            cfg = compose("config", overrides=[f"task={task}/mujoco"])
        override = EnvConfigAdapter(
            cfg, root_dir=root, algo_name="flashsac"
        ).build_task_env_cfg_override()
        env = registry.make(
            str(cfg.training.task_name),
            sim_backend="mujoco",
            num_envs=2,
            env_cfg_override=override,
        )
        try:
            obs, _ = env.reset()
            assert obs["obs"].shape[0] == 2
            assert obs["critic"].shape[0] == 2
            assert env.action_space.shape == (16,)

            env.reset(torch.tensor([0], dtype=torch.int64, device=env.device))
            for _ in range(3):
                state = env.step(torch.zeros((2, 16), device=env.device))
                assert torch.isfinite(state.reward).all()
                assert state.obs["obs"].shape == obs["obs"].shape
        finally:
            env.close()
