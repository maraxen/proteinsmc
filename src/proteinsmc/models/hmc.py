"""Data structures for HMC sampler."""

from __future__ import annotations

from dataclasses import dataclass

from proteinsmc.models.sampler_base import BaseSamplerConfig


@dataclass(frozen=True)
class HMCConfig(BaseSamplerConfig):
  """Configuration for the HMC sampler.

  Attributes:
      num_samples: The number of samples to generate.
      step_size: The step size for the integrator.
      num_integration_steps: The number of integration steps to take.

  """

  step_size: float = 0.1
  """Step size for the integrator."""
  num_integration_steps: int = 10
  """Number of integration steps to take."""
