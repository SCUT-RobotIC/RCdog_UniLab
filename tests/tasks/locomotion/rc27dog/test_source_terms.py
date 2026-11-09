"""Formula regressions against robot_lab's final wheeled RoughStones recipe."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unilab.managers.manager_base import ManagerTermBaseCfg
from unilab.managers.scene_entity_config import SceneEntityCfg
from unilab.tasks.locomotion.rc27dog.source_terms import (
    BasePositionZ,
    HeightScan,
    JointEffort,
    SourceReward,
)


class Scene(dict):
    read_plan = None


@pytest.fixture
def env():
    names = tuple(
        f"{leg}_{joint}_joint"
        for leg in ("FR", "FL", "RR", "RL")
        for joint in ("hip", "thigh", "calf")
    )
    names += tuple(f"{leg}_feet_joint" for leg in ("FR", "FL", "RR", "RL"))
    robot = SimpleNamespace(
        root_body_name="base_link",
        joint_names=names,
        data=SimpleNamespace(
            soft_joint_pos_limits=lambda _: torch.tensor([[-2.0, 4.0]]).repeat(16, 1)
        ),
    )
    root = SimpleNamespace(
        quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        pos_w=torch.tensor([[[0.0, 0.0, 0.72]]]),
        lin_vel_w=torch.zeros(1, 1, 3),
        ang_vel_w=torch.zeros(1, 1, 3),
    )
    joints = SimpleNamespace(joint_pos=torch.zeros(1, 16), joint_vel=torch.zeros(1, 16))
    sensors = {}
    plan = SimpleNamespace(
        body_tensor_view=lambda *_: root,
        joint_tensor_view=lambda *_: joints,
        sensor_tensor_views=lambda _, names: SimpleNamespace(
            values={name: sensors[name] for name in names}
        ),
    )
    scene = Scene(robot=robot)
    scene.read_plan = plan

    def scanner(**kwargs):
        return SimpleNamespace(
            scan=lambda: np.full((1, len(kwargs["offsets"])), 0.3, dtype=np.float32)
        )

    return SimpleNamespace(
        scene=scene,
        device="cpu",
        num_envs=1,
        physics_dt=0.005,
        common_step_counter=1,
        root=root,
        joints=joints,
        sensors=sensors,
        command=torch.zeros(1, 3),
        command_manager=SimpleNamespace(get_command=lambda _: torch.zeros(1, 3)),
        backend=SimpleNamespace(
            bind_height_scanner=scanner,
            get_geom_id=lambda _: 0,
            get_body_ids=lambda _: np.array([1]),
        ),
    )


def make_term(env, kind, **params):
    params = {"kind": kind, **params}
    return SourceReward(ManagerTermBaseCfg(func=SourceReward, params=params), env), params


def test_terrain_relative_height_uses_source_standing_height(env):
    reward, params = make_term(env, "base_height_l2", target_height=0.42)
    assert reward(env, **params).item() == pytest.approx(0.0, abs=1e-12)
    env.root.pos_w[..., 2] += 0.1
    assert reward(env, **params).item() == pytest.approx(0.01, abs=1e-7)


def test_vertical_velocity_is_in_body_frame_and_has_upright_gate(env):
    env.root.quat_w[:] = torch.tensor([[[0.9238795, 0.0, 0.3826834, 0.0]]])
    env.root.lin_vel_w[..., 0] = 2.0
    reward, params = make_term(env, "lin_vel_z_l2")
    assert reward(env, **params).item() == pytest.approx(2.0, rel=1e-5)


def test_velocity_limit_penalty_caps_each_violation_and_includes_wheels(env):
    env.joints.joint_vel[:, 0] = 35.0
    env.joints.joint_vel[:, 1] = 10.25
    env.joints.joint_vel[:, 2] = -10.5
    env.joints.joint_vel[:, 12] = 61.970496255240
    reward, params = make_term(env, "joint_vel_limits", soft_ratio=1.0)
    assert reward(env, **params).item() == pytest.approx(2.25, abs=1e-5)


def test_position_limits_shrink_authored_range_around_its_midpoint(env):
    cfg = SceneEntityCfg("robot")
    cfg.joint_ids = [0, 1]
    # Authored [-2, 4] becomes [-1.7, 3.7]; unselected joints must not contribute.
    env.joints.joint_pos[:, 0] = -1.8
    env.joints.joint_pos[:, 1] = 3.9
    env.joints.joint_pos[:, 2] = 100.0
    reward, params = make_term(env, "joint_pos_limits", asset_cfg=cfg)
    assert reward(env, **params).item() == pytest.approx(0.3, abs=1e-6)


def test_mirror_sums_each_three_joint_pair_before_averaging_pairs(env):
    env.joints.joint_pos[:, :3] = 1.0
    reward, params = make_term(
        env,
        "joint_mirror",
        mirror_joints=[
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ],
    )
    assert reward(env, **params).item() == pytest.approx(1.5)


def test_contact_force_excess_is_raw_and_has_no_upright_gate(env):
    env.root.quat_w[:] = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])
    env.sensors["contact_FR"] = torch.tensor([[0.0, 0.0, 100.0]])
    reward, params = make_term(env, "contact_forces", contact_names=["contact_FR"], threshold=80.0)
    assert reward(env, **params).item() == pytest.approx(20.0)


def test_stance_hip_uses_history_and_total_leg_count_denominator(env):
    names = [f"contact_{leg}" for leg in ("FR", "FL", "RR", "RL")]
    for name in names:
        env.sensors[name] = torch.zeros(1, 3)
    history = torch.zeros(1, 3, 4, 3)
    history[:, 0, 0, 2] = 3.0
    env.sensors["contact_history"] = history.flatten(1)
    cfg = SceneEntityCfg("robot")
    cfg.joint_ids = [0, 3, 6, 9]
    reward, params = make_term(
        env,
        "stance_hip_zero_reward",
        contact_names=names,
        history_contact_names=["contact_history"],
        asset_cfg=cfg,
        std=0.25,
    )
    assert reward(env, **params).item() == pytest.approx(0.25)


@pytest.mark.parametrize("kind,expected", [("contact_forces", 20.0), ("undesired_contacts", 1.0)])
def test_contact_rewards_include_transient_physics_step_contacts(env, kind, expected):
    names = ["contact_FR", "contact_FL"]
    for lag in range(3):
        for name in names:
            key = name if lag == 0 else f"{name}_lag{lag}"
            env.sensors[key] = torch.zeros(1, 3)
    env.sensors["contact_FR_lag1"][:, 2] = 100.0
    reward, params = make_term(env, kind, contact_names=names, history_length=3, threshold=80.0)
    assert reward.tensor_sensor_names == (
        "contact_FR",
        "contact_FL",
        "contact_FR_lag1",
        "contact_FL_lag1",
        "contact_FR_lag2",
        "contact_FL_lag2",
    )
    assert reward(env, **params).item() == pytest.approx(expected)


def test_acceleration_is_last_physics_step_velocity_difference_including_wheels(env):
    for name in env.scene["robot"].joint_names:
        env.sensors[f"joint_velocity_{name}"] = torch.ones(1, 1)
    env.joints.joint_vel[:] = 1.0
    env.joints.joint_vel[:, 0] += 0.01
    env.joints.joint_vel[:, 12] -= 0.015
    reward, params = make_term(env, "joint_acc_l2")
    assert all(name.startswith("joint_velocity_") for name in reward.tensor_sensor_names)
    assert reward(env, **params).item() == pytest.approx(13.0, rel=1e-5)


@pytest.mark.parametrize(
    "history,expected",
    [([1, 1, 0, 0, 1], 1.0), ([1, 1, 1, 1, 1], 0.0), ([0, 1, 0, 0, 0], 0.0)],
)
def test_first_contact_checks_all_physics_steps_in_control_period(env, history, expected):
    for lag, contact in enumerate(history):
        name = "contact_FR" if lag == 0 else f"contact_FR_lag{lag}"
        env.sensors[name] = torch.tensor([[0.0, 0.0, 3.0 * contact]])
    reward, params = make_term(
        env, "feet_contact_without_cmd", contact_names=["contact_FR"], history_length=5
    )
    reward._previous_contact = torch.tensor([[bool(history[-1])]])
    assert reward(env, **params).item() == pytest.approx(expected)


def test_first_contact_clears_previous_episode_sensor_history_on_reset(env):
    for lag in range(5):
        name = "contact_FR" if lag == 0 else f"contact_FR_lag{lag}"
        env.sensors[name] = torch.tensor([[0.0, 0.0, 3.0]])
    reward, params = make_term(
        env, "feet_contact_without_cmd", contact_names=["contact_FR"], history_length=5
    )
    assert reward(env, **params).item() == pytest.approx(1.0)
    env.common_step_counter += 1
    assert reward(env, **params).item() == pytest.approx(0.0)
    reward.reset(torch.tensor([0]))
    env.common_step_counter += 1
    assert reward(env, **params).item() == pytest.approx(1.0)


def test_contact_without_command_is_first_contact_not_persistent_support(env):
    env.sensors["contact_FR"] = torch.tensor([[0.0, 0.0, 3.0]])
    reward, params = make_term(env, "feet_contact_without_cmd", contact_names=["contact_FR"])
    assert reward(env, **params).item() == pytest.approx(1.0)
    assert reward(env, **params).item() == pytest.approx(1.0)
    env.common_step_counter += 1
    assert reward(env, **params).item() == pytest.approx(0.0)
    reward.reset()
    env.common_step_counter += 1
    assert reward(env, **params).item() == pytest.approx(1.0)


def test_privileged_observations_preserve_source_shapes_and_values(env):
    cfg = ManagerTermBaseCfg(func=HeightScan)
    assert BasePositionZ(cfg, env)(env).item() == pytest.approx(0.72)
    height = HeightScan(cfg, env)(env)
    assert height.shape == (1, 187)
    torch.testing.assert_close(height, torch.full((1, 187), -0.08))
    for i, name in enumerate(env.scene["robot"].joint_names):
        env.sensors[f"joint_torque_{name}"] = torch.tensor([[float(i)]])
    efforts = JointEffort(cfg, env)(env)
    torch.testing.assert_close(efforts, torch.arange(16.0)[None])
