"""Final Isaac Lab observation and command recipe, after wheeled overrides."""

from pathlib import Path

from omegaconf import OmegaConf

_OWNER = Path(__file__).parents[2] / "src/unilab/conf/flashsac/task/rc27dog_rough_stones/base.yaml"
_LEGS = [
    f"{leg}_{joint}_joint" for leg in ("FR", "FL", "RR", "RL") for joint in ("hip", "thigh", "calf")
]
_WHEELS = [f"{leg}_feet_joint" for leg in ("FR", "FL", "RR", "RL")]


def test_rough_stones_command_ranges_and_control_timing_match_source() -> None:
    owner = OmegaConf.load(_OWNER)
    assert owner.env.sim_dt == 0.005
    assert owner.env.ctrl_dt == 0.02
    assert owner.env.max_episode_seconds == 20.0
    command = owner.env.commands.twist
    assert command.resampling_time_range == [5.0, 15.0]
    assert command.rel_standing_envs == 0.02
    assert command.heading_command is True
    assert command.heading_control_stiffness == 0.5
    assert command.rel_heading_envs == 1.0
    assert command.ranges.lin_vel_x == [-0.8, 0.8]
    assert command.ranges.lin_vel_y == [-0.6, 0.6]
    assert command.ranges.ang_vel_z == [-0.8, 0.8]


def test_rough_stones_observations_keep_source_history_noise_and_scales() -> None:
    groups = OmegaConf.load(_OWNER).env.observations
    assert groups.policy.enable_corruption is True
    assert groups.critic.get("enable_corruption", False) is False
    expected = {
        "policy": [
            ("base_ang_vel", 0.25),
            ("projected_gravity", 1.0),
            ("command", 1.0),
            ("joint_pos", 1.0),
            ("joint_vel", 0.05),
            ("actions", 1.0),
        ],
        "critic": [
            ("base_lin_vel", 1.0),
            ("base_ang_vel", 1.0),
            ("base_pos_z", 1.0),
            ("projected_gravity", 1.0),
            ("command", 1.0),
            ("joint_pos", 1.0),
            ("joint_vel", 1.0),
            ("actions", 1.0),
            ("height_scan", 1.0),
            ("joint_effort", 0.01),
        ],
    }
    for group_name, sequence in expected.items():
        group = groups[group_name]
        terms = group.terms
        # Names of command terms are arbitrary; their position is part of policy I/O.
        actual_names = ["command" if name == "velocity_commands" else name for name in terms]
        assert actual_names == [name for name, _ in sequence]
        for (name, scale), term in zip(sequence, terms.values(), strict=True):
            assert term.get("scale", 1.0) == scale
            expected_history = 0 if name == "height_scan" else 5
            assert term.get("history_length", group.get("history_length", 0)) == expected_history
            assert term.get("clip") == ([-1.0, 1.0] if name == "height_scan" else [-100.0, 100.0])
    for name, magnitude in {
        "base_ang_vel": 0.2,
        "projected_gravity": 0.05,
        "joint_pos": 0.03,
        "joint_vel": 1.5,
    }.items():
        noise = groups.policy.terms[name].noise
        assert noise.n_min == -magnitude
        assert noise.n_max == magnitude
    for group in groups.values():
        assert group.terms.joint_pos.params.asset_cfg.joint_names == _LEGS
        assert group.terms.joint_pos.params.asset_cfg.preserve_order is True
        assert group.terms.joint_vel.params.asset_cfg.joint_names == _LEGS + _WHEELS
        assert group.terms.joint_vel.params.asset_cfg.preserve_order is True


def test_rough_stones_enabled_reward_count_and_weights_match_final_source() -> None:
    rewards = OmegaConf.load(_OWNER).reward
    # configure_wheeled_rc removes air-time/slide/feet-height terms after the
    # rough-stones class sets them, leaving precisely these 23 live weights.
    weights = sorted(
        float(term.weight) for term in rewards.values() if term is not None and term.weight != 0
    )
    assert weights == sorted(
        [
            -100,
            -1,
            -1,
            -0.2,
            -1.2,
            -6,
            -3e-5,
            -3e-6,
            -4,
            -3,
            -2.5e-5,
            -0.7,
            -0.8,
            -0.04,
            -0.02,
            -1.5,
            -2.5e-4,
            5.5,
            2.5,
            0.05,
            0.35,
            1,
            -1,
        ]
    )
