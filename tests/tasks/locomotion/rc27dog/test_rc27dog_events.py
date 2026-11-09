"""Source event semantics independently checked through the public entity seam."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from unilab.base.reset_selectors import ResetMaskBuffer
from unilab.managers.rng import ManagerRng
from unilab.managers.scene_entity_config import SceneEntityCfg
from unilab.tasks.locomotion.rc27dog.events import (
    PersistentExternalWrench,
    PushRootVelocity,
    RandomizeContactMaterial,
    RandomizeJointParameters,
    RandomizeMassInertia,
    ResetJointsByOffset,
    ResetRootStateUniform,
)


class Entity:
    def __init__(self):
        self.num_joints = 3
        self.root_body_name = "base_link"
        self.body_names = ("base_link",)
        self.writes: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.data = SimpleNamespace(
            default_joint_pos=lambda device: torch.tensor([[0.2, 0.8, 0.0]]).repeat(3, 1),
            default_joint_vel=lambda device: torch.zeros((3, 3)),
            default_root_state=lambda device: torch.tensor(
                [[0.0, 0.0, 0.42, 1.0, 0.0, 0.0, 0.0] + [0.0] * 6]
            ).repeat(3, 1),
            soft_joint_pos_limits=lambda device: torch.tensor(
                [[-1.0, 1.0], [-1.0, 1.0], [-torch.inf, torch.inf]]
            ),
        )

    def bind_body_mass_write(self, ids, **kwargs):
        return [0, 1], torch.tensor([2.0, 5.0])

    def bind_body_inertia_write(self, ids, **kwargs):
        return [0, 1], torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])

    def bind_joint_frictionloss_write(self, ids, **kwargs):
        return [0, 1], torch.tensor([0.0, 2.0])

    def bind_joint_armature_write(self, ids, **kwargs):
        return [0, 1], torch.tensor([1.0, 4.0])

    def bind_geom_friction_write(self, ids, **kwargs):
        return [0, 1], torch.tensor([[0.8, 0.005, 0.0001], [0.8, 0.005, 0.0001]])

    def bind_geom_solref_write(self, ids, **kwargs):
        return [0, 1], torch.tensor([[0.02, 1.0], [0.03, 1.0]])

    def bind_body_wrench(self, ids, **kwargs):
        return [0], [0]

    def bind_root_linear_velocity_delta(self, **kwargs):
        pass

    def bind_root_angular_velocity_delta(self, **kwargs):
        pass

    def __getattr__(self, name: str):
        if name.startswith("write_"):

            def write(values, columns, rows, **kwargs):
                self.writes[name] = (values.clone(), rows.clone())

            return write
        raise AttributeError(name)

    def write_joint_state_to_sim_mask(self, position, velocity, mask, columns):
        rows = mask.nonzero().flatten()
        self.writes["position"] = (position[rows].clone(), rows.clone())
        self.writes["velocity"] = (velocity[rows].clone(), rows.clone())

    def apply_body_wrench_to_sim(self, force, torque, columns, rows, **kwargs):
        self.writes["force"] = (force.clone(), rows.clone())
        self.writes["torque"] = (torque.clone(), rows.clone())

    def apply_root_velocity_delta_to_sim(self, linear, angular, rows, **kwargs):
        self.writes["linear_delta"] = (linear.clone(), rows.clone())
        self.writes["angular_delta"] = (angular.clone(), rows.clone())


class Scene(dict):
    pass


@pytest.fixture
def env():
    entity = Entity()
    scene = Scene(robot=entity)
    root = SimpleNamespace(
        quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]).repeat(3, 1, 1),
        lin_vel_w=torch.tensor([[[4.0, 5.0, 6.0]]]).repeat(3, 1, 1),
        ang_vel_w=torch.tensor([[[1.0, 2.0, 3.0]]]).repeat(3, 1, 1),
    )
    scene.read_plan = SimpleNamespace(body_tensor_view=lambda entity, names: root)
    return SimpleNamespace(
        num_envs=3,
        device=torch.device("cpu"),
        scene=scene,
        rng=ManagerRng(device="cpu", generator=torch.Generator().manual_seed(14)),
        event_term_mask_buffer=ResetMaskBuffer(3, "cpu"),
    )


def _term(cls: Any, env: Any, **params: Any):
    return cls(SimpleNamespace(mode="reset", params=params), env), params


@pytest.mark.parametrize(
    "operation,bounds,expected",
    [("add", (1.0, 1.0), [3.0, 6.0]), ("scale", (0.8, 0.8), [1.6, 4.0])],
)
def test_mass_inertia_scales_identically_and_replays_on_partial_reset(
    env, operation, bounds, expected
):
    term, params = _term(
        RandomizeMassInertia, env, operation=operation, mass_distribution_params=bounds
    )
    term(env, None, **params)
    writes = env.scene["robot"].writes
    full_mass = writes["write_body_mass_to_sim"][0]
    full_inertia = writes["write_body_inertia_to_sim"][0]
    torch.testing.assert_close(full_mass[0], torch.tensor(expected))
    torch.testing.assert_close(
        full_inertia / torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
        (full_mass / torch.tensor([2.0, 5.0]))[..., None].expand_as(full_inertia),
    )
    term(env, torch.tensor([2]), **params)
    torch.testing.assert_close(writes["write_body_mass_to_sim"][0], full_mass[2:3])
    torch.testing.assert_close(writes["write_body_inertia_to_sim"][0], full_inertia[2:3])


def test_joint_parameters_are_fixed_per_world_and_zero_friction_stays_zero(env):
    term, params = _term(
        RandomizeJointParameters,
        env,
        friction_distribution_params=(0.2, 1.5),
        armature_distribution_params=(0.8, 1.2),
    )
    term(env, None, **params)
    writes = env.scene["robot"].writes
    full_friction = writes["write_joint_frictionloss_to_sim"][0]
    full_armature = writes["write_joint_armature_to_sim"][0]
    assert torch.all(full_friction[:, 0] == 0)
    assert torch.all((full_friction[:, 1] >= 0.4) & (full_friction[:, 1] <= 3.0))
    assert torch.all(
        (full_armature / torch.tensor([1.0, 4.0]) >= 0.8)
        & (full_armature / torch.tensor([1.0, 4.0]) <= 1.2)
    )
    term(env, torch.tensor([0, 2]), **params)
    torch.testing.assert_close(writes["write_joint_frictionloss_to_sim"][0], full_friction[[0, 2]])
    torch.testing.assert_close(writes["write_joint_armature_to_sim"][0], full_armature[[0, 2]])


def test_leg_and_wheel_resets_use_distinct_offsets(env):
    leg, leg_params = _term(
        ResetJointsByOffset,
        env,
        asset_cfg=SceneEntityCfg("robot", joint_ids=[0, 1]),
        position_range=(-0.05, 0.05),
        velocity_range=(-0.5, 0.5),
    )
    leg(env, torch.tensor([1]), **leg_params)
    writes = env.scene["robot"].writes
    offset = writes["position"][0] - torch.tensor([[0.2, 0.8]])
    assert torch.all(offset.abs() <= 0.05)
    assert torch.all(writes["velocity"][0].abs() <= 0.5)
    wheel, wheel_params = _term(
        ResetJointsByOffset,
        env,
        asset_cfg=SceneEntityCfg("robot", joint_ids=[2]),
        position_range=(0.0, 0.0),
        velocity_range=(0.0, 0.0),
    )
    wheel(env, torch.tensor([1]), **wheel_params)
    assert torch.all(writes["position"][0] == 0)
    assert torch.all(writes["velocity"][0] == 0)
    torch.testing.assert_close(writes["position"][1], torch.tensor([1]))


def test_root_height_replaces_shared_keyframe_and_preserves_ranges(env, monkeypatch):
    captured = {}

    def reset(*args):
        captured["pose_range"] = args[2]
        captured["ids"] = args[1]

    monkeypatch.setattr("unilab.tasks.locomotion.rc27dog.events.reset_root_state_uniform", reset)
    term, params = _term(
        ResetRootStateUniform,
        env,
        initial_height=0.48,
        pose_range={"z": (0.0, 0.1), "roll": (-0.05, 0.05)},
    )
    term(env, torch.tensor([1]), **params)
    assert captured["pose_range"]["z"] == pytest.approx((0.06, 0.16))
    assert captured["pose_range"]["roll"] == (-0.05, 0.05)
    assert params["pose_range"]["z"] == (0.0, 0.1)
    torch.testing.assert_close(captured["ids"], torch.tensor([1]))


def test_material_preserves_rolling_friction_and_contact_time_constant(env):
    term, params = _term(
        RandomizeContactMaterial,
        env,
        static_friction_range=(0.35, 1.4),
        dynamic_friction_range=(0.7, 0.7),
        restitution_range=(0.2, 0.2),
        num_buckets=64,
    )
    term(env, None, **params)
    writes = env.scene["robot"].writes
    friction = writes["write_geom_friction_to_sim"][0]
    solref = writes["write_geom_solref_to_sim"][0]
    torch.testing.assert_close(friction[..., 0], torch.full((3, 2), 0.7))
    torch.testing.assert_close(friction[..., 1:], torch.tensor([0.005, 0.0001]).expand(3, 2, 2))
    torch.testing.assert_close(solref[..., 0], torch.tensor([0.02, 0.03]).expand(3, 2))
    damping = -math.log(0.2) / math.sqrt(math.pi**2 + math.log(0.2) ** 2)
    torch.testing.assert_close(solref[..., 1], torch.full((3, 2), damping))
    term(env, torch.tensor([2]), **params)
    torch.testing.assert_close(writes["write_geom_friction_to_sim"][0], friction[2:3])


def test_push_overwrites_requested_world_velocity_components(env):
    term, params = _term(
        PushRootVelocity,
        env,
        velocity_range={"x": (0.2, 0.2), "y": (-0.3, -0.3), "roll": (0.1, 0.1)},
    )
    term(env, torch.tensor([2]), **params)
    writes = env.scene["robot"].writes
    torch.testing.assert_close(writes["linear_delta"][0], torch.tensor([[-3.8, -5.3, 0.0]]))
    torch.testing.assert_close(writes["angular_delta"][0], torch.tensor([[-0.9, 0.0, 0.0]]))


def test_persistent_wrench_rotates_local_vectors_and_replays(env):
    term, params = _term(
        PersistentExternalWrench, env, force_range=(2.0, 2.0), torque_range=(3.0, 3.0)
    )
    term.reset(torch.tensor([1]))
    root = env.scene.read_plan.body_tensor_view(None, None)
    root.quat_w[1, 0] = torch.tensor([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
    term(env, None, **params)
    writes = env.scene["robot"].writes
    torch.testing.assert_close(writes["force"][0][1, 0], torch.tensor([-2.0, 2.0, 2.0]))
    torch.testing.assert_close(writes["torque"][0][1, 0], torch.tensor([-3.0, 3.0, 3.0]))
    assert torch.all(writes["force"][0][[0, 2]] == 0)
    first = writes["force"][0].clone()
    term(env, None, **params)
    torch.testing.assert_close(writes["force"][0], first)
