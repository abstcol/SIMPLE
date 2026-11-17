# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""Opt-in episode physics settings; task definitions and observations stay unchanged."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MassPerturbation:
    """Scale target mass and inertia at construction, preserving its mass distribution."""

    enabled: bool = False
    mass_scale: float = 1.0

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('physics.enabled must be a boolean')
        if (type(self.mass_scale) not in (int, float)
                or not math.isfinite(self.mass_scale) or self.mass_scale <= 0):
            raise ValueError('physics.mass_scale must be finite and positive')

    @property
    def effective_scale(self):
        return self.mass_scale if self.enabled else 1.0
