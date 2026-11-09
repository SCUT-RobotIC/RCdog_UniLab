"""RC terrain progression and map exits ported from IsaacLab's velocity task."""

from __future__ import annotations

from typing import Any

import torch

from unilab.envs.mdp.observations import root_body_view
from unilab.managers.scene_entity_config import SceneEntityCfg

_ROBOT = SceneEntityCfg("robot")


def terrain_levels_vel(
    env: Any,
    env_ids: torch.Tensor,
    command_name: str = "twist",
    tile_length: float = 8.0,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Advance after crossing half a tile; retreat after insufficient command travel.

    Reaching the hardest row resamples a row uniformly, matching the source
    TerrainImporter. Only resetting environments change their level and origin.
    """
    terrain = env.scene.terrain
    if not terrain.is_curriculum:
        raise NotImplementedError("terrain_levels_vel requires curriculum terrain")
    # The bootstrap reset runs before the root reset event has placed robots
    # at tile origins. Preserve max_init_terrain_level=0 until physics starts.
    if env.common_step_counter == 0:
        return terrain.terrain_levels.float().mean()
    _, root = root_body_view(env, asset_cfg, "terrain_levels_vel")
    distance = torch.linalg.vector_norm(
        root.pos_w[env_ids, 0, :2] - terrain.env_origins[env_ids, :2], dim=1
    )
    command = env.command_manager.get_command(command_name)
    if command is None:
        raise KeyError(f"command '{command_name}' is unavailable")
    move_up = distance > tile_length / 2.0
    move_down = (
        distance
        < torch.linalg.vector_norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    ) & ~move_up
    levels = terrain.terrain_levels[env_ids] + move_up.long() - move_down.long()
    rows = terrain.terrain_origins.shape[0]
    sampled_levels = env.rng.integers(0, rows, len(env_ids), dtype=torch.int64)
    terrain.select_levels(env_ids, torch.where(levels >= rows, sampled_levels, levels.clamp(min=0)))
    return terrain.terrain_levels.float().mean()


def terrain_out_of_bounds(
    env: Any,
    distance_buffer: float = 3.0,
    tile_size: tuple[float, float] = (8.0, 8.0),
    border_width: float = 20.0,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Time out at the generated map edge, including its flat border.

    UniSim's curriculum map has one column per terrain type. Read the actual
    generated row/column count so the limit follows the materialized map.
    """
    terrain = env.scene.terrain
    rows, columns = terrain.terrain_origins.shape[:2]
    _, root = root_body_view(env, asset_cfg, "terrain_out_of_bounds")
    limit_x = rows * tile_size[0] / 2.0 + border_width - distance_buffer
    limit_y = columns * tile_size[1] / 2.0 + border_width - distance_buffer
    return (root.pos_w[:, 0, 0].abs() > limit_x) | (root.pos_w[:, 0, 1].abs() > limit_y)


setattr(terrain_levels_vel, "reset_scoped", terrain_levels_vel)
setattr(terrain_out_of_bounds, "tensor_body_names", ("base_link",))


__all__ = ["terrain_levels_vel", "terrain_out_of_bounds"]
