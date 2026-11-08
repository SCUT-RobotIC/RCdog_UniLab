"""Manager terms used by the RC27Dog wheeled locomotion tasks.

The functions in the former IsaacLab task are kept behind two small dispatch
terms (``CommonReward`` and ``ObstacleReward``).  This keeps owner YAMLs
readable while making every state dependency explicit to UniLab's packed read
plan.  Torque and contact values are read from named public sensors; no
backend-private state is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from unilab.envs.mdp.commands.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from unilab.envs.mdp.observations import joint_view, require_read_plan, root_body_view
from unilab.managers.command_manager import CommandTerm, CommandTermCfg
from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from unilab.managers.scene_entity_config import SceneEntityCfg
from unilab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

_ROBOT = SceneEntityCfg("robot")


def _u(env: Any, quat: torch.Tensor) -> torch.Tensor:
    gravity = torch.zeros((env.num_envs, 3), device=quat.device, dtype=quat.dtype)
    gravity[:, 2] = -1.0
    projected = quat_apply_inverse(quat, gravity)
    return torch.clamp(-projected[:, 2], 0.0, 0.7) / 0.7


def _root(env: Any, cfg: SceneEntityCfg = _ROBOT):
    _, view = root_body_view(env, cfg, "rc27dog")
    return view


def _command(env: Any, name: str) -> torch.Tensor:
    value = env.command_manager.get_command(name)
    if value is None:
        raise KeyError(f"command '{name}' is unavailable")
    return value.to(device=env.device, dtype=torch.float32)


def _origins(env: Any) -> torch.Tensor:
    """Return per-environment origins through the public scene API."""
    return env.scene.env_origins(env.device)


def _joint(env: Any, cfg: SceneEntityCfg = _ROBOT):
    return joint_view(env, cfg, "rc27dog")


def _sensor(env: Any, names: tuple[str, ...]) -> torch.Tensor:
    plan = require_read_plan(env, "rc27dog")
    values = plan.sensor_tensor_views(env.scene["robot"], names).values
    return torch.cat([values[name].reshape(env.num_envs, -1) for name in names], dim=1)


def _selected(values: torch.Tensor, ids: list[int] | slice) -> torch.Tensor:
    return values[:, ids]


def _kind(term: str, kind: str) -> str:
    if not isinstance(kind, str) or not kind:
        raise ValueError(f"{term} kind must be a non-empty string")
    return kind


class CommonReward(ManagerTermBase):
    """Dispatch common RC rewards using the legacy function name in ``kind``."""

    _allowed_params: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, cfg: ManagerTermBaseCfg, env: Any):
        super().__init__(env)
        params = cfg.params
        kind = params.get("kind")
        names: list[str] = []
        asset_cfg = params.get("asset_cfg", _ROBOT)
        entity = (
            env.scene[asset_cfg.name]
            if isinstance(asset_cfg, SceneEntityCfg)
            else env.scene["robot"]
        )
        self._tensor_body_names = (entity.root_body_name,)
        if kind in {"joint_power", "joint_torques_l2"}:
            names.extend(f"joint_torque_{name}" for name in entity.joint_names)
        if kind == "joint_power":
            names.extend(f"joint_velocity_{name}" for name in entity.joint_names)
        if kind == "joint_acc_l2":
            names.extend(f"joint_acceleration_{name}" for name in entity.joint_names)
        for key in ("contact_names", "sensor_names"):
            value = params.get(key, ())
            if isinstance(value, (tuple, list)):
                names.extend(str(name) for name in value)
        self._tensor_sensor_names = tuple(dict.fromkeys(names))

    @property
    def tensor_sensor_names(self) -> tuple[str, ...]:
        return self._tensor_sensor_names

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return self._tensor_body_names

    def __call__(self, env: Any, kind: str, **params: Any) -> torch.Tensor:
        kind = _kind(self.name, kind)
        cfg = params.get("asset_cfg", _ROBOT)
        view = _root(env, cfg)
        quat = view.quat_w[:, 0]
        upright = _u(env, quat)
        cmd_name = params.get("command_name", "twist")
        cmd = _command(env, cmd_name)
        if kind in {"track_lin_vel_xy_exp", "track_lin_vel_xy_yaw_frame_exp"}:
            std = float(params["std"])
            actual = quat_apply_inverse(quat, view.lin_vel_w[:, 0])
            if kind == "track_lin_vel_xy_yaw_frame_exp":
                actual = quat_apply_inverse(yaw_quat(quat), view.lin_vel_w[:, 0])
            return torch.exp(-torch.sum((cmd[:, :2] - actual[:, :2]) ** 2, 1) / std**2) * upright
        if kind in {"body_x_alignment", "body_x_alignment_with_initial_yaw"}:
            forward = quat_apply(
                quat, torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, -1)
            )
            if kind.endswith("initial_yaw"):
                target = getattr(
                    env, "_cardinal_yaw_target", torch.zeros(env.num_envs, device=env.device)
                )
                alignment = forward[:, 0] * torch.cos(target) + forward[:, 1] * torch.sin(target)
                return (0.5 * (alignment.clamp(-1, 1) + 1.0)) * upright
            return (0.5 * (forward[:, 0].clamp(-1, 1) + 1.0)) * upright
        if kind in {"track_ang_vel_z_exp", "track_ang_vel_z_world_exp"}:
            std = float(params["std"])
            actual = (
                view.ang_vel_w[:, 0, 2]
                if kind.endswith("world_exp")
                else quat_apply_inverse(quat, view.ang_vel_w[:, 0])[:, 2]
            )
            return torch.exp(-((cmd[:, 2] - actual) ** 2) / std**2) * upright
        if kind == "lin_vel_z_l2":
            return view.lin_vel_w[:, 0, 2].square() * upright
        if kind == "ang_vel_xy_l2":
            actual = quat_apply_inverse(quat, view.ang_vel_w[:, 0])
            return actual[:, :2].square().sum(1) * upright
        if kind == "lin_vel_xy_delta_l2":
            actual = quat_apply_inverse(quat, view.lin_vel_w[:, 0])
            return (cmd[:, :2] - actual[:, :2]).square().sum(1) * upright
        if kind == "flat_orientation_l2":
            gravity = quat_apply_inverse(
                quat, torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, -1)
            )
            return gravity[:, :2].square().sum(1) * upright
        if kind == "upward":
            return (
                1.0
                - quat_apply_inverse(
                    quat, torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, -1)
                )[:, 2]
            ).square()
        if kind == "base_height_l2":
            target = float(params["target_height"])
            return (view.pos_w[:, 0, 2] - target).square() * upright
        if kind in {"stand_still", "joint_pos_penalty"}:
            joints = _selected(_joint(env, cfg).joint_pos, cfg.joint_ids)
            default = _selected(
                env.scene[cfg.name].data.default_joint_pos(env.device), cfg.joint_ids
            )
            error = (
                (joints - default).abs().sum(1)
                if kind == "stand_still"
                else torch.linalg.vector_norm(joints - default, dim=1)
            )
            if kind == "stand_still":
                return (
                    error
                    * (
                        torch.linalg.vector_norm(cmd, dim=1)
                        < float(params.get("command_threshold", 0.1))
                    ).float()
                    * upright
                )
            moving = (
                torch.linalg.vector_norm(cmd, dim=1) > float(params.get("command_threshold", 0.1))
            ) | (
                torch.linalg.vector_norm(
                    quat_apply_inverse(quat, view.lin_vel_w[:, 0])[:, :2], dim=1
                )
                > float(params.get("velocity_threshold", 0.2))
            )
            scale = torch.where(
                moving,
                torch.ones_like(error),
                torch.full_like(error, float(params["stand_still_scale"])),
            )
            return error * scale * upright
        if kind == "joint_pos_limits":
            pos = _selected(_joint(env, cfg).joint_pos, cfg.joint_ids)
            limits = env.scene[cfg.name].data.soft_joint_pos_limits(env.device)[cfg.joint_ids]
            return (
                torch.clamp(limits[None, :, 0] - pos, min=0)
                + torch.clamp(pos - limits[None, :, 1], min=0)
            ).sum(1)
        if kind == "joint_vel_limits":
            vel = _selected(_joint(env, cfg).joint_vel, cfg.joint_ids).abs()
            limits_fn = getattr(env.scene[cfg.name].data, "soft_joint_vel_limits", None)
            if limits_fn is None:
                raise NotImplementedError(
                    "RC27Dog joint_vel_limits requires public soft velocity limits"
                )
            limits = limits_fn(env.device)[cfg.joint_ids]
            return torch.clamp(vel - limits[None, :], min=0).sum(1)
        if kind in {"joint_power", "joint_torques_l2", "joint_acc_l2"}:
            names = tuple(
                f"joint_{'torque' if kind == 'joint_power' or kind == 'joint_torques_l2' else 'acceleration'}_{n}"
                for n in env.scene[cfg.name].joint_names
            )
            values = _selected(_sensor(env, names), cfg.joint_ids)
            if kind == "joint_power":
                qvel = _selected(_joint(env, cfg).joint_vel, cfg.joint_ids)
                return (values.abs() * qvel.abs()).sum(1)
            return values.square().sum(1)
        if kind == "action_rate_l2":
            return (env.action_manager.action - env.action_manager.prev_action).square().sum(1)
        if kind == "action_mirror":
            pairs = params.get("mirror_joints", ())
            action = env.action_manager.action
            return torch.stack(
                [(action[:, a].abs() - action[:, b].abs()).square() for a, b in pairs], 1
            ).mean(1)
        if kind == "action_sync":
            groups = params.get("joint_groups", ())
            vals = [action_abs(action=env.action_manager.action, ids=group) for group in groups]
            return (
                torch.stack([v.var(1, unbiased=False) for v in vals], 1).mean(1)
                if vals
                else torch.zeros(env.num_envs, device=env.device)
            )
        if kind == "joint_mirror":
            joints = _joint(env, cfg).joint_pos
            pairs = params.get("mirror_joints", ())
            return (
                torch.stack([(joints[:, a] - joints[:, b]).square() for a, b in pairs], 1).mean(1)
                * upright
                if pairs
                else torch.zeros(env.num_envs, device=env.device)
            )
        if kind == "undesired_contacts":
            names = tuple(params["contact_names"])
            force = _sensor(env, names).abs()
            return (force > float(params.get("threshold", 1.0))).sum(1).float() * upright
        if kind == "contact_forces":
            force = _sensor(env, tuple(params["contact_names"])).abs()
            return (
                torch.clamp(
                    (force - float(params["threshold"])) / float(params["max_force"]), min=0, max=1
                )
                .max(1)
                .values
            )
        if kind == "low_bar_clearance":
            bar = float(params["bar_height"])
            origins = _origins(env)
            near = (
                view.pos_w[:, 0, 0] - origins[:, 0] - float(params.get("bar_x", 1.0))
            ).abs() < float(params["approach_distance"])
            return (
                torch.clamp(view.pos_w[:, 0, 2] - (bar - float(params["margin"])), min=0.0) * near
            )
        raise NotImplementedError(f"RC27Dog common reward kind '{kind}' is not implemented")

    @property
    def tensor_state_entity(self) -> str:
        return "robot"


def action_abs(action: torch.Tensor, ids: list[int] | tuple[int, ...]) -> torch.Tensor:
    return action[:, list(ids)].abs()


class ObstacleReward(CommonReward):
    """Dispatch low-wall and broken-bridge geometric rewards."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: Any):
        super().__init__(cfg, env)
        names = list(self._tensor_body_names)
        for key in ("foot_body_names", "body_names"):
            value = cfg.params.get(key, ())
            if isinstance(value, (tuple, list)):
                names.extend(str(name) for name in value)
        self._tensor_body_names = tuple(dict.fromkeys(names))

    def __call__(self, env: Any, kind: str, **params: Any) -> torch.Tensor:
        if kind in {"low_wall_progress", "broken_bridge_progress"}:
            view = _root(env, params.get("asset_cfg", _ROBOT))
            origins = _origins(env)
            x = view.pos_w[:, 0, 0] - origins[:, 0]
            if kind == "low_wall_progress":
                wall, distance = float(params["wall_x"]), float(params["finish_distance"])
                value = (x - wall + distance) / (2 * distance)
            else:
                value = (x - float(params["start_x"])) / (
                    float(params["finish_x"]) - float(params["start_x"])
                )
            return torch.clamp(value, 0, 1) * _u(env, view.quat_w[:, 0])
        if kind == "broken_bridge_centerline":
            view = _root(env, params.get("asset_cfg", _ROBOT))
            std = float(params["std"])
            origins = _origins(env)
            y = view.pos_w[:, 0, 1] - origins[:, 1]
            return torch.exp(-y.square() / std**2) * _u(env, view.quat_w[:, 0])
        if kind == "broken_bridge_feet_above_groove":
            view = require_read_plan(env, "rc27dog").body_tensor_view(
                "robot", tuple(params["foot_body_names"])
            )
            origins = _origins(env)
            x = view.pos_w[:, :, 0] - origins[:, None, 0]
            z = view.pos_w[:, :, 2]
            flat, width, depth = (
                float(params["x_flat_length"]),
                float(params["groove_width"]),
                float(params["groove_depth"]),
            )
            active = torch.remainder(x + 0.5 * (flat + width), flat + width) >= flat
            return (
                active & (z > -depth + float(params.get("bottom_margin", 0.03))) & (z < 0)
            ).float().mean(1) * _u(env, _root(env).quat_w[:, 0])
        return super().__call__(env, kind, **params)


class ThresholdVelocityCommand(UniformVelocityCommand):
    """Uniform command with IsaacLab's 0.2 m/s dead band."""

    def _resample_command(self, reset_mask: torch.Tensor) -> None:
        super()._resample_command(reset_mask)
        self._tensor_command[:, :2].mul_(
            (torch.linalg.vector_norm(self._tensor_command[:, :2], dim=1) > 0.2)[:, None]
        )


@dataclass
class ThresholdVelocityCommandCfg(UniformVelocityCommandCfg):
    def build(self, env: Any) -> ThresholdVelocityCommand:
        return ThresholdVelocityCommand(self, env)


class BrokenBridgeVelocityCommand(ThresholdVelocityCommand):
    """Sample forward, turn-in-place, or stand-still bridge modes."""

    def _resample_command(self, reset_mask: torch.Tensor) -> None:
        super()._resample_command(reset_mask)
        ids = torch.nonzero(reset_mask).flatten()
        if ids.numel() == 0:
            return
        p = torch.rand(ids.numel(), device=self._device)
        f = float(getattr(self.cfg, "forward_only_prob", 0.7))
        t = f + float(getattr(self.cfg, "turn_in_place_prob", 0.25))
        self._tensor_command[ids] = 0
        forward = ids[p < f]
        turn = ids[(p >= f) & (p < t)]
        if forward.numel():
            self._tensor_command[forward, 0] = self._sample_uniform(
                self.cfg.ranges.lin_vel_x[0], self.cfg.ranges.lin_vel_x[1], forward.numel()
            )
        if turn.numel():
            self._tensor_command[turn, 2] = self._sample_uniform(
                self.cfg.ranges.ang_vel_z[0], self.cfg.ranges.ang_vel_z[1], turn.numel()
            )


@dataclass
class BrokenBridgeVelocityCommandCfg(ThresholdVelocityCommandCfg):
    forward_only_prob: float = 0.70
    turn_in_place_prob: float = 0.25
    standing_prob: float = 0.05
    min_turn_abs_ang_vel_z: float = 0.0

    def build(self, env: Any) -> BrokenBridgeVelocityCommand:
        return BrokenBridgeVelocityCommand(self, env)


class BaseHeightCommand(CommandTerm):
    def __init__(self, cfg: "BaseHeightCommandCfg", env: Any):
        super().__init__(cfg, env)
        self._base_height_cfg = cfg
        self._command = torch.zeros((self.num_envs, 1), device=self._device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, reset_mask: torch.Tensor) -> None:
        lo, hi = self._base_height_cfg.height_range
        self._command.copy_(
            torch.where(
                reset_mask[:, None],
                self._rng.uniform(lo, hi, (self.num_envs, 1), dtype=torch.float32),
                self._command,
            )
        )

    def _update_command(self) -> None:
        pass

    def _update_metrics(self) -> None:
        pass


@dataclass
class BaseHeightCommandCfg(CommandTermCfg):
    height_range: tuple[float, float] = (0.20, 0.28)

    def build(self, env: Any) -> BaseHeightCommand:
        return BaseHeightCommand(self, env)


def dead_down(env: Any, threshold: float = 0.5, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    view = _root(env, asset_cfg)
    gravity = quat_apply_inverse(
        view.quat_w[:, 0],
        torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, -1),
    )
    return gravity[:, 2] > threshold


setattr(dead_down, "tensor_body_names", ("base_link",))


__all__ = [
    "CommonReward",
    "ObstacleReward",
    "ThresholdVelocityCommandCfg",
    "BrokenBridgeVelocityCommandCfg",
    "BaseHeightCommandCfg",
    "dead_down",
]
