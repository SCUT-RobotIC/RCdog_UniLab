"""Hydra owner checks for the RC27Dog wheel-legged task family."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

_TASKS = (
    "rc27dog_flat",
    "rc27dog_rough",
    "rc27dog_rough_stones",
    "rc27dog_low_bar",
    "rc27dog_low_wall",
    "rc27dog_broken_bridge",
)


def _compose(task: str):
    config_dir = Path(__file__).parents[2] / "src" / "unilab" / "conf" / "flashsac"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose("config", overrides=[f"task={task}/mujoco"])


def test_all_rc27dog_mujoco_owners_compose() -> None:
    for task in _TASKS:
        cfg = _compose(task)
        assert (
            cfg.training.task_name
            == {
                "rc27dog_flat": "RC27DogWheelFlat",
                "rc27dog_rough": "RC27DogWheelRough",
                "rc27dog_rough_stones": "RC27DogWheelRoughStones",
                "rc27dog_low_bar": "RC27DogWheelLowBar",
                "rc27dog_low_wall": "RC27DogWheelLowWall",
                "rc27dog_broken_bridge": "RC27DogWheelBrokenBridge",
            }[task]
        )
        assert cfg.training.sim_backend == "mujoco"
        assert cfg.reward.track_lin_vel.func == "unilab.envs.mdp.track_linear_velocity"
        if task in {"rc27dog_low_bar", "rc27dog_low_wall", "rc27dog_broken_bridge"}:
            assert cfg.env.commands.twist.heading_command is False
            assert cfg.env.commands.twist.ranges.heading is None


def test_low_bar_and_broken_bridge_rewards_are_top_level_manager_terms() -> None:
    low_bar = _compose("rc27dog_low_bar")
    assert "reward" not in low_bar.env
    assert low_bar.reward.base_height is None

    bridge = _compose("rc27dog_broken_bridge")
    assert bridge.reward.progress.func.endswith("manager_terms.ObstacleReward")
    assert bridge.reward.progress.params.kind == "broken_bridge_progress"
