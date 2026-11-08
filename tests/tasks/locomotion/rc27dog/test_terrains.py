from __future__ import annotations

from pathlib import Path

import numpy as np

from unilab.tasks.locomotion.rc27dog.terrains import (
    StonyRoadTerrainCfg,
    rough_stones_terrain,
    rough_terrain,
)


def test_rc_rough_terrain_preserves_source_mix() -> None:
    cfg = rough_terrain()
    assert [cfg.sub_terrains[name].proportion for name in cfg.sub_terrains] == [0.4, 0.4, 0.2]
    assert cfg.curriculum is True


def test_stony_road_has_edge_and_diagonal_ridges() -> None:
    cfg = StonyRoadTerrainCfg(
        size=(2.0, 2.0),
        noise_range=(0.0, 0.0),
        horizontal_scale=0.05,
        vertical_scale=0.005,
    )
    output = cfg.function(0.0, np.random.default_rng(0))
    heights = output.heightfield.physical_heights_xy()
    assert heights[0, 10] >= 0.15
    assert heights[-1, 10] >= 0.15
    assert heights[10, 10] >= 0.05
    assert output.heightfield.base_thickness == 0.1


def test_stony_road_mix_contains_all_seven_source_types() -> None:
    cfg = rough_stones_terrain()
    assert set(cfg.sub_terrains) == {
        "flat_stony",
        "light_stony_rough",
        "hard_stony_rough",
        "pyramid_stairs",
        "pyramid_stairs_inv",
        "pyramid_slope",
        "pyramid_slope_inv",
    }


def test_rc_obstacle_fragments_are_authored_collision_geometry() -> None:
    root = Path(__file__).parents[4] / "src" / "unilab" / "assets" / "robots" / "rc27dog"
    low_bar = (root / "low_bar.xml").read_text()
    low_wall = (root / "low_wall.xml").read_text()
    bridge = (root / "scene_broken_bridge.xml").read_text()
    assert 'type="cylinder"' in low_bar
    assert low_wall.count('type="box"') == 16
    assert 'pos="0 0 -0.2"' in bridge
    assert "bridge_block_" in bridge
