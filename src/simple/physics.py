# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""Validate optional episode physics without importing simulator dependencies."""

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class MassPerturbation:
    """Scale one actor's mass and inertia at construction."""

    enabled: bool = False
    mass_scale: float = 1.0
    actor: str = 'target'

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('physics.enabled must be a boolean')
        if self.actor not in ('target', 'articulated'):
            raise ValueError('physics.mass.actor must be target or articulated')
        scale = self.mass_scale
        try:
            valid = type(scale) in (int, float) and math.isfinite(scale) and scale > 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError('physics mass scale must be finite and positive')

    @property
    def effective_scale(self):
        return self.mass_scale if self.enabled else 1.0


@dataclass(frozen=True)
class PhysicsConfig:
    """Per-episode settings. An omitted perturbation uses nominal physics."""

    mass: MassPerturbation = field(default_factory=MassPerturbation)

    @classmethod
    def from_dict(cls, config: dict | None) -> 'PhysicsConfig':
        if config is None:
            return cls()
        if not isinstance(config, dict):
            raise ValueError('physics must be a configuration object')

        unknown = config.keys() - {'mass', 'enabled', 'mass_scale'}
        if unknown:
            names = ', '.join(sorted(map(str, unknown)))
            raise ValueError(f'Unknown physics settings: {names}')
        if 'mass' not in config:
            # Preserve old manifests, including mass_scale without enabled.
            return cls(mass=MassPerturbation(**config))
        if 'enabled' in config or 'mass_scale' in config:
            raise ValueError('physics.mass cannot be combined with legacy enabled or mass_scale')
        mass = config['mass']
        if not isinstance(mass, dict) or 'scale' not in mass or mass.keys() - {'scale', 'actor'}:
            raise ValueError('physics.mass requires scale and accepts an optional actor')
        return cls(mass=MassPerturbation(enabled=True, mass_scale=mass['scale'],
                                         actor=mass.get('actor', 'target')))
