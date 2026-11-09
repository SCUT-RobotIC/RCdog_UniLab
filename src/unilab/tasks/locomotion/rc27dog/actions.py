"""RC leg servos with the source actuator's episode-held physical-step delay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from unilab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from unilab.managers._buffers import CircularBuffer


@dataclass(kw_only=True)
class DelayedLegActionCfg(JointPositionActionCfg):
    """Position targets; one independent delay per env and hip/thigh/calf group."""

    min_delay: int = 2
    max_delay: int = 7

    def build(self, env: Any) -> DelayedLegAction:
        return DelayedLegAction(self, env)


class DelayedLegAction(JointPositionAction):
    def __init__(self, cfg: DelayedLegActionCfg, env: Any):
        super().__init__(cfg, env)
        self.delay_cfg = cfg
        if not 0 <= cfg.min_delay <= cfg.max_delay:
            raise ValueError("RC leg delays require 0 <= min_delay <= max_delay")
        if cfg.delay_max_lag:
            raise ValueError("RC actions use min_delay/max_delay for grouped delays")
        names = self._target_names
        if len(names) != 12 or any(name.endswith("feet_joint") for name in names):
            raise ValueError("DelayedLegAction requires exactly the 12 RC leg joints")
        self.requires_substep_state_feedback = True
        self.groups = torch.tensor(
            [("hip", "thigh", "calf").index(name.split("_")[1]) for name in names],
            device=env.device,
        )
        self.rng = env.rng
        self.history = CircularBuffer(max_len=cfg.max_delay + 1, batch_size=env.num_envs)
        self.lags = torch.zeros((env.num_envs, 3), dtype=torch.long, device=env.device)
        self.rows = torch.arange(env.num_envs, device=env.device)[:, None]
        self.columns = torch.arange(12, device=env.device)[None, :]

    def _delay_target(self, target: torch.Tensor) -> torch.Tensor:
        self.history.append(target)
        # CircularBuffer returns oldest-to-newest history. Each actuator group
        # holds its sampled lag until reset, matching Isaac DelayedPDActuator.
        available = self.history.current_length - 1
        lag = torch.minimum(self.lags[:, self.groups], available[:, None])
        history = self.history.buffer
        return history[self.rows, -lag - 1, self.columns]

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        ids = torch.arange(self.num_envs, device=self._device) if env_ids is None else env_ids
        self.history.reset(ids)
        self.lags[ids] = (
            self.rng.uniform(
                float(self.delay_cfg.min_delay),
                float(self.delay_cfg.max_delay + 1),
                (ids.numel(), 3),
            )
            .long()
            .clamp_max(self.delay_cfg.max_delay)
        )
