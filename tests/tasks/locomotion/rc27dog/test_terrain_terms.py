"""Source curriculum thresholds, subset resets, and generated-map bounds."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from unilab.managers.rng import ManagerRng
from unilab.tasks.locomotion.rc27dog.terrain_terms import (
    terrain_levels_vel,
    terrain_out_of_bounds,
)


class Terrain:
    def __init__(self):
        self.is_curriculum = True
        self.terrain_origins = torch.zeros((10, 7, 3))
        self.terrain_origins[:, :, 0] = torch.arange(10)[:, None] * 8 - 36
        self.terrain_origins[:, :, 1] = torch.arange(7)[None, :] * 8 - 24
        self.terrain_levels = torch.tensor([2, 2, 2, 2])
        self.terrain_types = torch.tensor([0, 1, 2, 3])
        self.env_origins = self.terrain_origins[self.terrain_levels, self.terrain_types].clone()

    def select_levels(self, ids, levels):
        self.terrain_levels[ids] = levels
        self.env_origins[ids] = self.terrain_origins[levels, self.terrain_types[ids]]


class Scene(dict):
    pass


def environment():
    terrain = Terrain()
    root = SimpleNamespace(pos_w=terrain.env_origins[:, None, :].clone())
    command = torch.tensor([[1.0, 0.0, 0.0]]).repeat(4, 1)
    scene = Scene(robot=SimpleNamespace(root_body_name="base_link"))
    scene.terrain = terrain
    scene.read_plan = SimpleNamespace(body_tensor_view=lambda entity, names: root)
    env = SimpleNamespace(
        scene=scene,
        rng=ManagerRng.seeded(7, device="cpu"),
        max_episode_length_s=20.0,
        common_step_counter=1,
        command_manager=SimpleNamespace(get_command=lambda name: command),
    )
    return env, terrain, root, command


def test_distance_curriculum_uses_source_thresholds_and_up_precedence():
    env, terrain, root, command = environment()
    root.pos_w[:, 0, 0] += torch.tensor([4.01, 4.0, 2.0, 1.99])
    command[2:, 0] = 0.2
    result = terrain_levels_vel(env, torch.arange(4))
    assert terrain.terrain_levels.tolist() == [3, 1, 2, 1]
    assert result.item() == 1.75


def test_curriculum_changes_only_reset_rows_and_updates_spawn_origins():
    env, terrain, root, _ = environment()
    previous_origins = terrain.env_origins.clone()
    root.pos_w[1, 0, 1] += 5.0
    terrain_levels_vel(env, torch.tensor([1]))
    assert terrain.terrain_levels.tolist() == [2, 3, 2, 2]
    torch.testing.assert_close(terrain.env_origins[[0, 2, 3]], previous_origins[[0, 2, 3]])
    torch.testing.assert_close(terrain.env_origins[1], terrain.terrain_origins[3, 1])


def test_curriculum_bottom_clamps_and_top_resamples_reproducibly():
    env, terrain, root, _ = environment()
    terrain.terrain_levels[:] = torch.tensor([0, 9, 9, 9])
    root.pos_w[1:, 0, 0] += 5.0
    expected_rng = ManagerRng.seeded(7, device="cpu")
    expected = expected_rng.integers(0, 10, 4)
    terrain_levels_vel(env, torch.arange(4))
    assert terrain.terrain_levels[0].item() == 0
    torch.testing.assert_close(terrain.terrain_levels[1:], expected[1:])
    assert torch.all((terrain.terrain_levels >= 0) & (terrain.terrain_levels < 10))


def test_out_of_bounds_follows_actual_map_with_source_buffer():
    env, _, root, _ = environment()
    # Actual 10x7 map: half extents (40,28), border20, inward buffer3.
    root.pos_w[:, 0, :2] = torch.tensor([[57.0, 45.0], [57.01, 0.0], [0.0, -45.01], [-57.0, -45.0]])
    assert terrain_out_of_bounds(env).tolist() == [False, True, True, False]


def test_empty_reset_does_not_change_curriculum():
    env, terrain, _, _ = environment()
    initial = terrain.env_origins.clone()
    terrain_levels_vel(env, torch.empty(0, dtype=torch.int64))
    torch.testing.assert_close(terrain.env_origins, initial)
    assert terrain.terrain_levels.tolist() == [2, 2, 2, 2]


def test_bootstrap_reset_keeps_start_level_before_robot_placement():
    env, terrain, root, _ = environment()
    env.common_step_counter = 0
    terrain.terrain_levels.zero_()
    root.pos_w.zero_()
    terrain_levels_vel(env, torch.arange(4))
    assert terrain.terrain_levels.tolist() == [0, 0, 0, 0]
