"""Physical-step delay semantics for the three source leg servo groups."""

from types import SimpleNamespace

import torch

from unilab.managers.rng import ManagerRng
from unilab.tasks.locomotion.rc27dog.actions import DelayedLegActionCfg


def test_leg_groups_delay_physical_targets_and_partial_reset_independently():
    names = [
        f"{leg}_{joint}_joint"
        for leg in ("FR", "FL", "RR", "RL")
        for joint in ("hip", "thigh", "calf")
    ]
    entity = SimpleNamespace(
        find_joints_by_actuator_names=lambda _: (list(range(12)), names),
        num_joints=16,
        data=SimpleNamespace(default_joint_pos=lambda device: torch.zeros((2, 16))),
    )
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene={"robot": entity},
        rng=ManagerRng(device="cpu", generator=torch.Generator().manual_seed(42)),
    )
    action = DelayedLegActionCfg(entity_name="robot", actuator_names=names).build(env)
    assert action.requires_substep_state_feedback
    action.reset()
    assert torch.all((action.lags >= 2) & (action.lags <= 7))
    action.lags[:] = torch.tensor([[2, 4, 7], [3, 5, 6]])
    held_lags = action.lags.clone()
    for physical_step in range(12):
        target = torch.full((2, 12), float(physical_step))
        delayed = action._delay_target(target)
        expected = (physical_step - held_lags[:, action.groups]).clamp_min(0).float()
        torch.testing.assert_close(delayed, expected)
        torch.testing.assert_close(action.lags, held_lags)
    action.reset(torch.tensor([0]))
    torch.testing.assert_close(action.lags[1], held_lags[1])
    delayed = action._delay_target(torch.full((2, 12), 12.0))
    torch.testing.assert_close(delayed[0], torch.full((12,), 12.0))
    torch.testing.assert_close(delayed[1], (12 - held_lags[1, action.groups]).float())
