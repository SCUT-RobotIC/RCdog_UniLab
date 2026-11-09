"""Hydra owner checks for the RC27Dog wheel-legged task family."""

from __future__ import annotations

from pathlib import Path
from shutil import copytree

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

_CONFIG_DIR = Path(__file__).parents[2] / "src" / "unilab" / "conf" / "flashsac"

_TASKS = (
    "rc27dog_flat",
    "rc27dog_rough",
    "rc27dog_rough_stones",
    "rc27dog_low_bar",
    "rc27dog_low_wall",
    "rc27dog_broken_bridge",
)


def _compose(task: str, backend: str = "mujoco", *, config_dir: Path = _CONFIG_DIR):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose("config", overrides=[f"task={task}/{backend}"])


@pytest.mark.parametrize("backend", ("mujoco", "mjwarp"))
def test_all_rc27dog_owners_compose(backend: str) -> None:
    for task in _TASKS:
        cfg = _compose(task, backend)
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
        assert cfg.training.sim_backend == backend
        if task == "rc27dog_rough_stones":
            assert cfg.reward.track_lin_vel.func.endswith("source_terms.SourceReward")
            assert cfg.reward.track_lin_vel.params.kind == "track_lin_vel_xy_exp"
        else:
            assert cfg.reward.track_lin_vel.func == "unilab.envs.mdp.track_linear_velocity"
        if task in {"rc27dog_low_bar", "rc27dog_low_wall", "rc27dog_broken_bridge"}:
            assert cfg.env.commands.twist.heading_command is False
            assert cfg.env.commands.twist.ranges.heading is None


@pytest.mark.parametrize("task", _TASKS)
@pytest.mark.parametrize("backend", ("mujoco", "mjwarp"))
def test_task_tuning_does_not_change_other_tasks(task: str, backend: str, tmp_path: Path) -> None:
    config_dir = copytree(_CONFIG_DIR, tmp_path / "flashsac")
    before = {name: _compose(name, backend, config_dir=config_dir) for name in _TASKS}
    owner_path = config_dir / "task" / task / "base.yaml"
    owner = OmegaConf.load(owner_path)
    changes = {
        "env.commands.twist.ranges.lin_vel_x": [0.125, 0.25],
        "env.actions.wheel_vel.scale": 7.0,
        "env.observations.policy.terms.base_ang_vel.scale": 0.75,
        "env.events.reset_root_state_uniform.params.pose_range.x": [-0.1, 0.1],
        "env.terminations.bad_orientation.params.limit_angle": 0.75,
        "reward.track_lin_vel.weight": 17.0,
    }
    for key, value in changes.items():
        OmegaConf.update(owner, key, value)
    owner_path.write_text("# @package _global_\n" + OmegaConf.to_yaml(owner), encoding="utf-8")

    for name in _TASKS:
        after = _compose(name, backend, config_dir=config_dir)
        if name == task:
            for key, value in changes.items():
                assert OmegaConf.select(after, key) == value
        else:
            assert after == before[name], f"Tuning {task} changed {name}/{backend}"


def test_low_bar_and_broken_bridge_rewards_are_top_level_manager_terms() -> None:
    low_bar = _compose("rc27dog_low_bar")
    assert "reward" not in low_bar.env
    assert low_bar.reward.base_height is None

    bridge = _compose("rc27dog_broken_bridge")
    assert bridge.reward.progress.func.endswith("manager_terms.ObstacleReward")
    assert bridge.reward.progress.params.kind == "broken_bridge_progress"
