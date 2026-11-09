"""RC wheel-legged task events matching the Isaac Lab training recipe.

Physical randomization is sampled once per world and replayed on reset. This
keeps Isaac Lab's startup semantics through UniLab's public reset transaction.
MuJoCo has one sliding friction coefficient, so material randomization uses
the source dynamic coefficient; restitution is approximated by solref damping.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from unilab.envs.mdp.events import reset_root_state_uniform, resolve_env_ids
from unilab.envs.mdp.observations import root_body_view
from unilab.managers.manager_base import ManagerTermBase
from unilab.managers.scene_entity_config import SceneEntityCfg
from unilab.utils.math import quat_apply

_ROBOT = SceneEntityCfg("robot")


def _range(value: Any) -> tuple[float, float]:
    lower, upper = map(float, value)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise ValueError(f"Invalid uniform range {value!r}")
    return lower, upper


def _sample(env: Any, bounds: Any, shape: tuple[int, ...]) -> torch.Tensor:
    return env.rng.uniform(*_range(bounds), shape, dtype=torch.float32)


def _defaults(values: Any, count: int, rank: int, device: Any) -> torch.Tensor:
    table = (
        values.to(device=device, dtype=torch.float32)
        if isinstance(values, torch.Tensor)
        else torch.tensor(values, device=device, dtype=torch.float32)
    )
    return table.expand(count, *table.shape).clone() if table.ndim == rank else table.clone()


class ResetRootStateUniform(ManagerTermBase):
    """Use the task's initial root height independently of the shared keyframe."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(env)
        asset_cfg = cfg.params.get("asset_cfg", _ROBOT)
        defaults = env.scene[asset_cfg.name].data.default_root_state(env.device)
        initial_height = float(cfg.params["initial_height"])
        if not math.isfinite(initial_height):
            raise ValueError("Initial root height must be finite")
        if not torch.allclose(defaults[:, 2], defaults[0, 2].expand(env.num_envs)):
            raise ValueError("RC root reset requires the same authored height per world")
        self._height_offset = initial_height - float(defaults[0, 2])

    def __call__(
        self,
        env: Any,
        env_ids: torch.Tensor | None,
        pose_range: dict[str, tuple[float, float]],
        initial_height: float,
        velocity_range: dict[str, tuple[float, float]] | None = None,
        asset_cfg: SceneEntityCfg = _ROBOT,
    ) -> None:
        del initial_height
        ranges = dict(pose_range)
        lower, upper = _range(ranges.get("z", (0.0, 0.0)))
        ranges["z"] = (lower + self._height_offset, upper + self._height_offset)
        reset_root_state_uniform(
            env, resolve_env_ids(env, env_ids), ranges, velocity_range, asset_cfg
        )


class ResetJointsByOffset(ManagerTermBase):
    """Reset selected joints with additive position and velocity offsets."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(env)
        asset_cfg = cfg.params.get("asset_cfg", _ROBOT)
        self._entity = env.scene[asset_cfg.name]
        ids = asset_cfg.joint_ids
        self._joint_ids = (
            list(range(self._entity.num_joints))[ids] if isinstance(ids, slice) else list(ids)
        )
        self._positions = self._entity.data.default_joint_pos(env.device)[
            :, self._joint_ids
        ].clone()
        self._velocities = self._entity.data.default_joint_vel(env.device)[
            :, self._joint_ids
        ].clone()
        self._limits = _defaults(
            self._entity.data.soft_joint_pos_limits(env.device), env.num_envs, 2, env.device
        )[:, self._joint_ids]
        _range(cfg.params["position_range"])
        _range(cfg.params["velocity_range"])

    def __call__(
        self,
        env: Any,
        env_ids: torch.Tensor | None,
        position_range: tuple[float, float],
        velocity_range: tuple[float, float],
        asset_cfg: SceneEntityCfg = _ROBOT,
    ) -> None:
        del asset_cfg
        ids = resolve_env_ids(env, env_ids).to(device=env.device)
        shape = (env.num_envs, len(self._joint_ids))
        position = self._positions + _sample(env, position_range, shape)
        limits = self._limits
        position = position.clamp(min=limits[..., 0], max=limits[..., 1])
        velocity = self._velocities + _sample(env, velocity_range, shape)
        reset_mask = env.event_term_mask_buffer.scatter(ids)
        self._entity.write_joint_state_to_sim_mask(position, velocity, reset_mask, self._joint_ids)


class RandomizeMassInertia(ManagerTermBase):
    """Uniform startup mass randomization with proportional principal inertia."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(env)
        asset_cfg = cfg.params.get("asset_cfg", _ROBOT)
        self._entity = env.scene[asset_cfg.name]
        self._body_ids, mass = self._entity.bind_body_mass_write(
            asset_cfg.body_ids, term_name=self.name
        )
        inertia_ids, inertia = self._entity.bind_body_inertia_write(
            asset_cfg.body_ids, term_name=self.name
        )
        if list(self._body_ids) != list(inertia_ids):
            raise ValueError("Mass and inertia bindings must select the same bodies")
        self._mass = _defaults(mass, env.num_envs, 1, env.device)
        self._inertia = _defaults(inertia, env.num_envs, 2, env.device)
        self._operation = cfg.params.get("operation", "scale")
        if self._operation not in {"add", "scale"}:
            raise ValueError("RC mass operation must be add or scale")
        self._range = _range(cfg.params["mass_distribution_params"])
        self._values: tuple[torch.Tensor, torch.Tensor] | None = None

    def __call__(
        self,
        env: Any,
        env_ids: torch.Tensor | None,
        mass_distribution_params: tuple[float, float],
        operation: str = "scale",
        asset_cfg: SceneEntityCfg = _ROBOT,
    ) -> None:
        del mass_distribution_params, operation, asset_cfg
        if self._values is None:
            samples = _sample(env, self._range, tuple(self._mass.shape))
            mass = self._mass + samples if self._operation == "add" else self._mass * samples
            if bool((mass <= 0).any()):
                raise ValueError("RC mass randomization produced a non-positive mass")
            inertia = self._inertia * (mass / self._mass)[..., None]
            self._values = mass, inertia
        ids = resolve_env_ids(env, env_ids).to(device=env.device)
        mass, inertia = self._values
        self._entity.write_body_mass_to_sim(mass[ids], self._body_ids, ids, term_name=self.name)
        self._entity.write_body_inertia_to_sim(
            inertia[ids], self._body_ids, ids, term_name=self.name
        )


class RandomizeJointParameters(ManagerTermBase):
    """Hold independently sampled joint friction and armature scales per world."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(env)
        asset_cfg = cfg.params.get("asset_cfg", _ROBOT)
        self._entity = env.scene[asset_cfg.name]
        self._joint_ids, friction = self._entity.bind_joint_frictionloss_write(
            asset_cfg.joint_ids, term_name=self.name
        )
        armature_ids, armature = self._entity.bind_joint_armature_write(
            asset_cfg.joint_ids, term_name=self.name
        )
        if list(self._joint_ids) != list(armature_ids):
            raise ValueError("Joint friction and armature bindings differ")
        self._friction = _defaults(friction, env.num_envs, 1, env.device)
        self._armature = _defaults(armature, env.num_envs, 1, env.device)
        self._friction_range = _range(cfg.params["friction_distribution_params"])
        self._armature_range = _range(cfg.params["armature_distribution_params"])
        self._values: tuple[torch.Tensor, torch.Tensor] | None = None

    def __call__(
        self,
        env: Any,
        env_ids: torch.Tensor | None,
        friction_distribution_params: tuple[float, float],
        armature_distribution_params: tuple[float, float],
        asset_cfg: SceneEntityCfg = _ROBOT,
    ) -> None:
        del friction_distribution_params, armature_distribution_params, asset_cfg
        if self._values is None:
            self._values = (
                self._friction * _sample(env, self._friction_range, tuple(self._friction.shape)),
                self._armature * _sample(env, self._armature_range, tuple(self._armature.shape)),
            )
        ids = resolve_env_ids(env, env_ids).to(device=env.device)
        friction, armature = self._values
        self._entity.write_joint_frictionloss_to_sim(
            friction[ids], self._joint_ids, ids, term_name=self.name
        )
        self._entity.write_joint_armature_to_sim(
            armature[ids], self._joint_ids, ids, term_name=self.name
        )


class RandomizeContactMaterial(ManagerTermBase):
    """Discrete source materials mapped to MuJoCo sliding friction and damping.

    Static friction is sampled to retain the source bucket recipe, but MuJoCo
    does not expose a separate static Coulomb coefficient. The dynamic sample
    therefore owns sliding friction; torsional and rolling friction stay at the
    authored robot values. solref time constants are preserved.
    """

    def __init__(self, cfg: Any, env: Any):
        super().__init__(env)
        asset_cfg = cfg.params.get("asset_cfg", _ROBOT)
        self._entity = env.scene[asset_cfg.name]
        self._geom_ids, friction = self._entity.bind_geom_friction_write(
            asset_cfg.geom_ids, term_name=self.name
        )
        solref_ids, solref = self._entity.bind_geom_solref_write(
            asset_cfg.geom_ids, term_name=self.name
        )
        if list(self._geom_ids) != list(solref_ids):
            raise ValueError("Friction and solref bindings differ")
        self._friction = _defaults(friction, env.num_envs, 2, env.device)
        self._solref = _defaults(solref, env.num_envs, 2, env.device)
        if bool((self._solref <= 0).any()):
            raise ValueError("RC material randomization requires positive solref format")
        self._ranges = [
            _range(cfg.params[key])
            for key in ("static_friction_range", "dynamic_friction_range", "restitution_range")
        ]
        if (
            self._ranges[0][0] < 0
            or self._ranges[1][0] < 0
            or not 0 <= self._ranges[2][0] <= self._ranges[2][1] < 1
        ):
            raise ValueError(
                "Material friction must be nonnegative and restitution must be in [0, 1)"
            )
        self._num_buckets = int(cfg.params.get("num_buckets", 64))
        if self._num_buckets <= 0:
            raise ValueError("num_buckets must be positive")
        self._values: tuple[torch.Tensor, torch.Tensor] | None = None

    def __call__(
        self,
        env: Any,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int = 64,
        asset_cfg: SceneEntityCfg = _ROBOT,
    ) -> None:
        del static_friction_range, dynamic_friction_range, restitution_range, num_buckets, asset_cfg
        if self._values is None:
            buckets = torch.stack(
                [_sample(env, bounds, (self._num_buckets,)) for bounds in self._ranges], dim=-1
            )
            samples = (
                _sample(env, (0.0, float(self._num_buckets)), tuple(self._friction.shape[:2]))
                .long()
                .clamp_max(self._num_buckets - 1)
            )
            materials = buckets[samples]
            friction, solref = self._friction.clone(), self._solref.clone()
            friction[..., 0] = materials[..., 1]
            log_restitution = materials[..., 2].clamp_min(1e-12).log()
            damping = -log_restitution / torch.sqrt(math.pi**2 + log_restitution.square())
            solref[..., 1] = torch.where(materials[..., 2] == 0, 1.0, damping)
            self._values = friction, solref
        ids = resolve_env_ids(env, env_ids).to(device=env.device)
        friction, solref = self._values
        self._entity.write_geom_friction_to_sim(
            friction[ids], self._geom_ids, ids, term_name=self.name
        )
        self._entity.write_geom_solref_to_sim(solref[ids], self._geom_ids, ids, term_name=self.name)


class PersistentExternalWrench(ManagerTermBase):
    """Reset-sampled body-local force and torque held for the whole episode.

    Isaac Lab's apply_external_force_torque stores a persistent local wrench.
    Re-stage it each control step through UniLab's public interval capability.
    """

    def __init__(self, cfg: Any, env: Any):
        super().__init__(env)
        self._asset_cfg = cfg.params.get("asset_cfg", _ROBOT)
        self._entity = env.scene[self._asset_cfg.name]
        self._local_ids, self._body_ids = self._entity.bind_body_wrench(
            self._asset_cfg.body_ids, torque=True, term_name=self.name
        )
        self._body_names = tuple(self._entity.body_names[int(i)] for i in self._local_ids)
        if self._body_names != (self._entity.root_body_name,):
            raise ValueError("RC external wrench must select the root body")
        self._force_range = _range(cfg.params["force_range"])
        self._torque_range = _range(cfg.params["torque_range"])
        self._forces = torch.zeros((env.num_envs, 1, 3), device=env.device)
        self._torques = torch.zeros_like(self._forces)

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return self._body_names

    @property
    def entity_name(self) -> str:
        return str(self._asset_cfg.name)

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, float]:
        ids = resolve_env_ids(self._env, env_ids).to(device=self._env.device)
        shape = (ids.numel(), 1, 3)
        self._forces[ids] = _sample(self._env, self._force_range, shape)
        self._torques[ids] = _sample(self._env, self._torque_range, shape)
        return {}

    def __call__(
        self,
        env: Any,
        env_ids: torch.Tensor | None,
        force_range: tuple[float, float],
        torque_range: tuple[float, float],
        asset_cfg: SceneEntityCfg = _ROBOT,
    ) -> None:
        del force_range, torque_range, asset_cfg
        ids = resolve_env_ids(env, env_ids).to(device=env.device)
        _, root = root_body_view(env, self._asset_cfg, self.name)
        quat = root.quat_w[ids, :1]
        self._entity.apply_body_wrench_to_sim(
            quat_apply(quat, self._forces[ids]),
            quat_apply(quat, self._torques[ids]),
            self._body_ids,
            ids,
            term_name=self.name,
        )


class PushRootVelocity(ManagerTermBase):
    """Overwrite sampled world velocity components, as Isaac Lab's push does."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(env)
        self._asset_cfg = cfg.params.get("asset_cfg", _ROBOT)
        self._entity = env.scene[self._asset_cfg.name]
        self._ranges = cfg.params["velocity_range"]
        unknown = set(self._ranges) - {"x", "y", "z", "roll", "pitch", "yaw"}
        if unknown:
            raise ValueError(f"Unknown velocity axes: {sorted(unknown)}")
        for bounds in self._ranges.values():
            _range(bounds)
        self._entity.bind_root_linear_velocity_delta(term_name=self.name)
        self._angular = bool(set(self._ranges) & {"roll", "pitch", "yaw"})
        if self._angular:
            self._entity.bind_root_angular_velocity_delta(term_name=self.name)

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return (self._entity.root_body_name,)

    @property
    def entity_name(self) -> str:
        return str(self._asset_cfg.name)

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, float]:
        del env_ids
        return {}

    def __call__(
        self,
        env: Any,
        env_ids: torch.Tensor | None,
        velocity_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg = _ROBOT,
    ) -> None:
        del asset_cfg
        ids = resolve_env_ids(env, env_ids).to(device=env.device)
        _, root = root_body_view(env, self._asset_cfg, self.name)
        current = torch.cat((root.lin_vel_w[ids, 0], root.ang_vel_w[ids, 0]), dim=-1)
        delta = torch.zeros_like(current)
        for index, key in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
            if key in velocity_range:
                delta[:, index] = (
                    _sample(env, velocity_range[key], (ids.numel(),)) - current[:, index]
                )
        self._entity.apply_root_velocity_delta_to_sim(
            delta[:, :3],
            delta[:, 3:] if self._angular else None,
            ids,
            term_name=self.name,
        )
