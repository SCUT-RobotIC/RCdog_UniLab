"""Cold-path terrain recipes for the RC wheeled tasks.

The stone road preserves the source height samples and diagonal ridges. Stairs
use UniSim heightfields, whose cell-wide risers differ from IsaacLab meshes.
Low walls and bridge grooves instead use authored box collision fragments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from unisim.terrain.generator import (
    SubTerrainCfg,
    TerrainGeneratorCfg,
    TerrainHeightField,
    TerrainOutput,
)

from unilab.terrains.heightfield_terrains import (
    HfFlatTerrainCfg,
    HfInvertedPyramidStairsTerrainCfg,
    HfPyramidStairsTerrainCfg,
)


def heightfield_output(
    heights: np.ndarray,
    *,
    size: tuple[float, float],
    horizontal_scale: float,
    vertical_scale: float,
    base_depth: float,
    spawn_height: float,
) -> TerrainOutput:
    """Encode physical heights without normalizing away the ground level."""
    noise = np.rint(heights / vertical_scale).astype(np.int16)
    elevation_min = int(noise.min())
    elevation_max = int(noise.max())
    return TerrainOutput(
        origin=np.array([size[0] / 2, size[1] / 2, spawn_height]),
        heightfield=TerrainHeightField(
            noise=noise,
            size=size,
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            elevation_min=elevation_min,
            elevation_max=elevation_max,
            max_physical_height=(elevation_max - elevation_min) * vertical_scale,
            base_thickness=base_depth,
            z_offset=elevation_min * vertical_scale,
        ),
    )


@dataclass(kw_only=True)
class StonyRoadTerrainCfg(SubTerrainCfg):
    """RC stone samples with raised tile edges and both diagonal ridges."""

    noise_range: tuple[float, float] = (0.0, 0.0)
    noise_step: float = 0.005
    downsampled_scale: float | None = None
    edge_wall_height: float = 0.15
    diagonal_wall_height: float = 0.05
    wall_thickness: float = 0.05
    base_depth: float = 0.1
    horizontal_scale: float = 0.05
    vertical_scale: float = 0.005

    def function(self, difficulty: float, rng: np.random.Generator) -> TerrainOutput:
        del difficulty
        scale = self.downsampled_scale or self.horizontal_scale
        if scale < self.horizontal_scale:
            raise ValueError("downsampled_scale must be >= horizontal_scale")
        if self.noise_step <= 0 or self.base_depth <= 0 or self.wall_thickness <= 0:
            raise ValueError("noise_step, base_depth and wall_thickness must be positive")
        if self.noise_range[0] > self.noise_range[1]:
            raise ValueError("noise_range must be ordered")
        shape = tuple(int(round(length / self.horizontal_scale)) for length in self.size)
        coarse_shape = tuple(max(1, int(length / scale)) for length in self.size)
        step = max(int(self.noise_step / self.vertical_scale), 1)
        values = np.arange(
            int(self.noise_range[0] / self.vertical_scale),
            int(self.noise_range[1] / self.vertical_scale) + step,
            step,
        )
        heights = rng.choice(values, size=coarse_shape)
        for axis in range(2):
            heights = heights.repeat(math.ceil(shape[axis] / coarse_shape[axis]), axis=axis)
        heights = heights[: shape[0], : shape[1]].astype(np.float64) * self.vertical_scale
        width = max(1, math.ceil(self.wall_thickness / self.horizontal_scale))
        heights[:width, :] = np.maximum(heights[:width, :], self.edge_wall_height)
        heights[-width:, :] = np.maximum(heights[-width:, :], self.edge_wall_height)
        heights[:, :width] = np.maximum(heights[:, :width], self.edge_wall_height)
        heights[:, -width:] = np.maximum(heights[:, -width:], self.edge_wall_height)
        rows, cols = np.ogrid[: shape[0], : shape[1]]
        main = abs(rows * (shape[1] - 1) - cols * (shape[0] - 1))
        anti = abs(rows * (shape[1] - 1) + cols * (shape[0] - 1) - (shape[0] - 1) * (shape[1] - 1))
        threshold = width * max(shape[0] - 1, shape[1] - 1)
        heights[(main <= threshold) | (anti <= threshold)] += self.diagonal_wall_height
        center = tuple(
            slice(
                max(0, n // 2 - int(1 / self.horizontal_scale)),
                n // 2 + int(1 / self.horizontal_scale),
            )
            for n in shape
        )
        return heightfield_output(
            heights,
            size=self.size,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            base_depth=self.base_depth,
            spawn_height=float(heights[center].max()),
        )


@dataclass(kw_only=True)
class PyramidSlopeTerrainCfg(SubTerrainCfg):
    """Four planar 20-degree faces and a central platform, as in robot_lab."""

    slope_angle: float = 20.0
    platform_width: float = 1.0
    border_width: float = 1.0
    inverted: bool = False
    horizontal_scale: float = 0.05
    vertical_scale: float = 0.005

    def function(self, difficulty: float, rng: np.random.Generator) -> TerrainOutput:
        del difficulty, rng
        if not 0 <= self.slope_angle < 90:
            raise ValueError("slope_angle must be in [0, 90) degrees")
        x, y = (
            np.arange(int(round(length / self.horizontal_scale))) * self.horizontal_scale
            for length in self.size
        )
        distance = np.minimum(
            np.minimum(x, self.size[0] - x)[:, None],
            np.minimum(y, self.size[1] - y)[None, :],
        )
        ramp_length = (min(self.size) - self.platform_width) / 2 - self.border_width
        if ramp_length <= 0:
            raise ValueError("platform and border leave no slope")
        heights = np.clip(distance - self.border_width, 0, ramp_length)
        heights *= math.tan(math.radians(self.slope_angle))
        if self.inverted:
            heights *= -1
        return heightfield_output(
            heights,
            size=self.size,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            base_depth=0.1,
            spawn_height=float(heights[len(x) // 2, len(y) // 2]),
        )


def rough_terrain() -> TerrainGeneratorCfg:
    """Source plane/stairs/inverted-stairs proportions (0.4/0.4/0.2)."""
    return TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=10,
        num_cols=20,
        curriculum=True,
        horizontal_scale=0.05,
        vertical_scale=0.005,
        sub_terrains={
            "plane": HfFlatTerrainCfg(proportion=0.4),
            "pyramid_stairs": HfPyramidStairsTerrainCfg(
                proportion=0.4,
                step_height_range=(0.0, 0.15),
                step_width=0.3,
                platform_width=1.0,
                border_width=1.0,
            ),
            "pyramid_stairs_inv": HfInvertedPyramidStairsTerrainCfg(
                proportion=0.2,
                step_height_range=(0.0, 0.15),
                step_width=0.3,
                platform_width=1.0,
                border_width=1.0,
            ),
        },
    )


def rough_stones_terrain() -> TerrainGeneratorCfg:
    """The seven source stone-road terrain types and spawning proportions."""
    return TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=10,
        num_cols=20,
        curriculum=True,
        horizontal_scale=0.05,
        vertical_scale=0.005,
        sub_terrains={
            "flat_stony": StonyRoadTerrainCfg(proportion=0.25),
            "light_stony_rough": StonyRoadTerrainCfg(
                proportion=0.30, noise_range=(0.01, 0.05), noise_step=0.01
            ),
            "hard_stony_rough": StonyRoadTerrainCfg(
                proportion=0.15, noise_range=(0.05, 0.12), noise_step=0.02
            ),
            "pyramid_stairs": HfPyramidStairsTerrainCfg(
                proportion=0.10,
                step_height_range=(0.1, 0.1),
                step_width=0.3,
                platform_width=1.0,
                border_width=1.0,
            ),
            "pyramid_stairs_inv": HfInvertedPyramidStairsTerrainCfg(
                proportion=0.10,
                step_height_range=(0.1, 0.1),
                step_width=0.3,
                platform_width=1.0,
                border_width=1.0,
            ),
            "pyramid_slope": PyramidSlopeTerrainCfg(proportion=0.05),
            "pyramid_slope_inv": PyramidSlopeTerrainCfg(proportion=0.05, inverted=True),
        },
    )


def _rough_sub_terrains() -> dict[str, SubTerrainCfg]:
    return {
        "plane": HfFlatTerrainCfg(proportion=0.4),
        "pyramid_stairs": HfPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.0, 0.15),
            step_width=0.3,
            platform_width=1.0,
            border_width=1.0,
        ),
        "pyramid_stairs_inv": HfInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.0, 0.15),
            step_width=0.3,
            platform_width=1.0,
            border_width=1.0,
        ),
    }


def _rough_stones_sub_terrains() -> dict[str, SubTerrainCfg]:
    return {
        "flat_stony": StonyRoadTerrainCfg(proportion=0.25),
        "light_stony_rough": StonyRoadTerrainCfg(
            proportion=0.30, noise_range=(0.01, 0.05), noise_step=0.01
        ),
        "hard_stony_rough": StonyRoadTerrainCfg(
            proportion=0.15, noise_range=(0.05, 0.12), noise_step=0.02
        ),
        "pyramid_stairs": HfPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.1, 0.1),
            step_width=0.3,
            platform_width=1.0,
            border_width=1.0,
        ),
        "pyramid_stairs_inv": HfInvertedPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.1, 0.1),
            step_width=0.3,
            platform_width=1.0,
            border_width=1.0,
        ),
        "pyramid_slope": PyramidSlopeTerrainCfg(proportion=0.05),
        "pyramid_slope_inv": PyramidSlopeTerrainCfg(proportion=0.05, inverted=True),
    }


@dataclass(kw_only=True)
class Rc27DogRoughTerrainCfg(TerrainGeneratorCfg):
    """Hydra materializable owner for the mixed stair terrain."""

    size: tuple[float, float] = (8.0, 8.0)
    border_width: float = 20.0
    num_rows: int = 10
    num_cols: int = 20
    curriculum: bool = True
    horizontal_scale: float = 0.05
    vertical_scale: float = 0.005
    sub_terrains: dict[str, SubTerrainCfg] = field(default_factory=_rough_sub_terrains)


@dataclass(kw_only=True)
class Rc27DogRoughStonesTerrainCfg(TerrainGeneratorCfg):
    """Hydra materializable owner for the stone-road terrain."""

    size: tuple[float, float] = (8.0, 8.0)
    border_width: float = 20.0
    num_rows: int = 10
    num_cols: int = 20
    curriculum: bool = True
    horizontal_scale: float = 0.05
    vertical_scale: float = 0.005
    sub_terrains: dict[str, SubTerrainCfg] = field(default_factory=_rough_stones_sub_terrains)
