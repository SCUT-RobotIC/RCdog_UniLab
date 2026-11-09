"""Reward and privileged-observation semantics of the source RC wheel task.

The source recipe applies ``configure_wheeled_rc`` last: its terrain-relative
standing height is 0.42 m and its swing-foot rewards are removed. This module
implements the remaining terms through public scene/backend capabilities.
"""

from __future__ import annotations

import re
from typing import Any

import torch

from unilab.envs.mdp.observations import joint_view, require_read_plan, root_body_view
from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from unilab.managers.scene_entity_config import SceneEntityCfg
from unilab.tasks.locomotion.rc27dog.manager_terms import CommonReward
from unilab.utils.math import quat_apply_inverse

_ROBOT = SceneEntityCfg("robot")


def _root(env: Any):
    return root_body_view(env, _ROBOT, "rc27dog_source")[1]


def _upright(env: Any) -> torch.Tensor:
    quat = _root(env).quat_w[:, 0]
    gravity = quat.new_tensor([0.0, 0.0, -1.0]).expand(env.num_envs, -1)
    return (-quat_apply_inverse(quat, gravity)[:, 2]).clamp(0.0, 0.7) / 0.7


def _sensors(env: Any, names: tuple[str, ...]) -> torch.Tensor:
    views = require_read_plan(env, "rc27dog_source").sensor_tensor_views(env.scene["robot"], names)
    return torch.cat([views.values[name].reshape(env.num_envs, -1) for name in names], dim=1)


class BasePositionZ(ManagerTermBase):
    """Source critic's world-frame base height, before history stacking."""

    entity_name = "robot"

    def __init__(self, cfg: ManagerTermBaseCfg, env: Any):
        super().__init__(env)
        self.tensor_body_names = (env.scene["robot"].root_body_name,)

    def __call__(self, env: Any) -> torch.Tensor:
        if env.scene.read_plan is None:
            return torch.zeros((env.num_envs, 1), device=env.device)
        return _root(env).pos_w[:, 0, 2:3]


class JointEffort(ManagerTermBase):
    """Applied actuator torque in the explicitly declared joint order."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: Any):
        super().__init__(env)
        self.tensor_sensor_names = tuple(
            f"joint_torque_{name}" for name in env.scene["robot"].joint_names
        )

    def __call__(self, env: Any) -> torch.Tensor:
        if env.scene.read_plan is None:
            return torch.zeros((env.num_envs, len(self.tensor_sensor_names)), device=env.device)
        return _sensors(env, self.tensor_sensor_names)


class HeightScan(ManagerTermBase):
    """Source yaw-aligned grid: base z minus terrain z minus 0.5 m."""

    entity_name = "robot"

    def __init__(self, cfg: ManagerTermBaseCfg, env: Any):
        super().__init__(env)
        self.tensor_body_names = (env.scene["robot"].root_body_name,)
        size = cfg.params.get("size", (1.6, 1.0))
        resolution = float(cfg.params.get("resolution", 0.1))
        x = torch.linspace(
            -size[0] / 2, size[0] / 2, round(size[0] / resolution) + 1, dtype=torch.float64
        )
        y = torch.linspace(
            -size[1] / 2, size[1] / 2, round(size[1] / resolution) + 1, dtype=torch.float64
        )
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        offsets = torch.stack((xx.flatten(), yy.flatten()), dim=-1).tolist()
        self.num_points = len(offsets)
        self.scanner = env.backend.bind_height_scanner(
            hfield_geom_id=env.backend.get_geom_id("floor"),
            offsets=offsets,
            frame_body_id=int(env.backend.get_body_ids(self.tensor_body_names)[0]),
            alignment="yaw",
            output="height",
        )

    def heights(self, env: Any) -> torch.Tensor:
        """Terrain world z at the bound grid points."""
        return torch.as_tensor(self.scanner.scan(), device=env.device, dtype=torch.float32)

    def __call__(
        self,
        env: Any,
        size: tuple[float, float] = (1.6, 1.0),
        resolution: float = 0.1,
        offset: float = 0.5,
    ) -> torch.Tensor:
        del size, resolution
        if env.scene.read_plan is None:
            return torch.zeros((env.num_envs, self.num_points), device=env.device)
        return _root(env).pos_w[:, 0, 2:3] - self.heights(env) - offset


class SourceReward(CommonReward):
    """Source formulas, including upright gates and unnormalized force excess.

    ``contact_names`` select vector net-force sensors in corresponding body
    order. ``history_length`` selects authored delayed sensors at each physical
    step. ``history_contact_names`` can supply an explicit flattened history.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: Any):
        super().__init__(cfg, env)
        self.kind = cfg.params["kind"]
        asset_cfg = cfg.params.get("asset_cfg", _ROBOT)
        entity = env.scene[asset_cfg.name]
        self._joint_position_limits = None
        self._joint_velocity_limits = None
        if self.kind == "joint_pos_limits":
            # EntityData exposes authored ranges; IsaacLab shrinks them by 0.9.
            limits = entity.data.soft_joint_pos_limits(env.device)[asset_cfg.joint_ids]
            center = limits.mean(dim=-1)
            half_width = (limits[:, 1] - limits[:, 0]) * 0.45
            self._joint_position_limits = torch.stack(
                (center - half_width, center + half_width), dim=-1
            )
        if self.kind == "joint_vel_limits":
            self._joint_velocity_limits = torch.tensor(
                [
                    61.470496255240
                    if name.endswith("_feet_joint")
                    else 30.0
                    if name.endswith("_hip_joint")
                    else 10.0
                    for name in entity.joint_names
                ],
                device=env.device,
            )[asset_cfg.joint_ids]
        self.history_contact_names = tuple(cfg.params.get("history_contact_names", ()))
        history_length = int(cfg.params.get("history_length", 1))
        if not self.history_contact_names and history_length > 1:
            self.history_contact_names = tuple(
                name if lag == 0 else f"{name}_lag{lag}"
                for lag in range(history_length)
                for name in cfg.params.get("contact_names", ())
            )
        if self.kind == "joint_acc_l2":
            # Authored velocity-stage sensors hold pre-integration qvel; the
            # packed joint view holds post-integration qvel. Their difference
            # matches IsaacLab's last-physics-step finite difference.
            self._tensor_sensor_names = tuple(
                f"joint_velocity_{name}" for name in entity.joint_names
            )
        self._tensor_sensor_names = tuple(
            dict.fromkeys((*self._tensor_sensor_names, *self.history_contact_names))
        )
        self.height_scan = None
        if self.kind == "base_height_l2":
            self.height_scan = HeightScan(
                ManagerTermBaseCfg(
                    func=HeightScan, params={"size": (0.1, 0.1), "resolution": 0.05}
                ),
                env,
            )
        self.mirror_pairs = []
        for patterns in cfg.params.get("mirror_joints", ()):
            self.mirror_pairs.append(
                tuple(
                    [
                        i
                        for i, name in enumerate(env.scene["robot"].joint_names)
                        if re.fullmatch(pattern, name)
                    ]
                    for pattern in patterns
                )
            )
        self._previous_contact: torch.Tensor | None = None
        self._first_contact: torch.Tensor | None = None
        self._contact_step: int | None = None

    def _forces(self, env: Any, names: tuple[str, ...]) -> torch.Tensor:
        return _sensors(env, names).reshape(env.num_envs, len(names), 3)

    def _history_forces(self, env: Any, params: dict[str, Any]) -> torch.Tensor:
        names = tuple(params["contact_names"])
        if self.history_contact_names:
            return _sensors(env, self.history_contact_names).reshape(
                env.num_envs, -1, len(names), 3
            )
        return self._forces(env, names)[:, None]

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if self._previous_contact is not None:
            rows = slice(None) if env_ids is None else env_ids
            self._previous_contact[rows] = False
            if self._first_contact is not None:
                self._first_contact[rows] = False

    def __call__(self, env: Any, kind: str, **params: Any) -> torch.Tensor:
        cfg = params.get("asset_cfg", _ROBOT)
        if kind == "joint_acc_l2":
            velocity = joint_view(env, cfg, "rc27dog_source").joint_vel
            previous = _sensors(env, self._tensor_sensor_names)
            acceleration = (velocity - previous) / env.physics_dt
            return acceleration[:, cfg.joint_ids].square().sum(1)
        if kind == "lin_vel_z_l2":
            root = _root(env)
            return quat_apply_inverse(root.quat_w[:, 0], root.lin_vel_w[:, 0])[
                :, 2
            ].square() * _upright(env)
        if kind == "base_height_l2":
            assert self.height_scan is not None
            heights = self.height_scan.heights(env)
            root_z = _root(env).pos_w[:, 0, 2]
            valid = torch.isfinite(heights).all() & (heights.abs().max() <= 1e6)
            target = torch.where(valid, float(params["target_height"]) + heights.mean(1), root_z)
            return (root_z - target).square() * _upright(env)
        if kind == "joint_pos_limits":
            positions = joint_view(env, cfg, "rc27dog_source").joint_pos[:, cfg.joint_ids]
            assert self._joint_position_limits is not None
            limits = self._joint_position_limits
            return (
                (limits[None, :, 0] - positions).clamp(min=0.0)
                + (positions - limits[None, :, 1]).clamp(min=0.0)
            ).sum(1)
        if kind == "joint_vel_limits":
            velocity = joint_view(env, cfg, "rc27dog_source").joint_vel[:, cfg.joint_ids]
            assert self._joint_velocity_limits is not None
            limits = self._joint_velocity_limits
            # IsaacLab bounds each joint's violation at 1 rad/s.
            return (
                (velocity.abs() - limits * float(params.get("soft_ratio", 1.0)))
                .clamp(0.0, 1.0)
                .sum(1)
            )
        if kind == "joint_mirror":
            positions = joint_view(env, cfg, "rc27dog_source").joint_pos
            values = [
                (positions[:, a] - positions[:, b]).square().sum(1) for a, b in self.mirror_pairs
            ]
            return torch.stack(values, 1).mean(1) * _upright(env)
        if kind in {
            "undesired_contacts",
            "contact_forces",
            "stance_hip_zero_reward",
            "stance_calf_singularity_penalty",
        }:
            forces = self._history_forces(env, params).norm(dim=-1).max(1).values
            threshold = float(params.get("threshold", params.get("contact_force_threshold", 1.0)))
            if kind == "undesired_contacts":
                return (forces > threshold).sum(1).float() * _upright(env)
            if kind == "contact_forces":
                return (forces - threshold).clamp(min=0.0).sum(1)
            positions = joint_view(env, cfg, "rc27dog_source").joint_pos[:, cfg.joint_ids]
            if positions.shape != forces.shape:
                raise ValueError("Stance reward requires one ordered joint per contact sensor")
            center = 0.0 if kind == "stance_hip_zero_reward" else float(params["singularity_angle"])
            return (
                torch.exp(-((positions - center) / float(params["std"])).square())
                * (forces > threshold)
            ).mean(1)
        if kind == "feet_contact_without_cmd":
            history = self._history_forces(env, params).norm(dim=-1) > 1.0
            contacts = history[:, 0]
            step = env.common_step_counter
            if self._previous_contact is None:
                self._previous_contact = torch.zeros_like(contacts)
            if step != self._contact_step:
                persistent = self._previous_contact
                if self.history_contact_names:
                    # The final delayed sample is the previous control boundary.
                    # Track that boundary explicitly so per-env reset clears it
                    # even if the simulator retains pre-reset sensor history.
                    persistent = persistent & history[:, 1:-1].all(dim=1)
                self._first_contact = contacts & ~persistent
                self._previous_contact.copy_(contacts)
                self._contact_step = step
            assert self._first_contact is not None
            command = env.command_manager.get_command(params.get("command_name", "twist"))
            return self._first_contact.sum(1).float() * (command.norm(dim=1) < 0.1) * _upright(env)
        return super().__call__(env, kind=kind, **params)
