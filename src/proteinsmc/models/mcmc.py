"""Data structures for MCMC sampler."""

from __future__ import annotations

from dataclasses import dataclass

from proteinsmc.models.sampler_base import BaseSamplerConfig

DEFAULT_STEP_SIZE = 1e-1


@dataclass(frozen=True)
class MCMCConfig(BaseSamplerConfig):
  """Configuration for the MCMC sampler using a Random Walk Metropolis kernel.

  Attributes:
      step_size: The step size (standard deviation of the Gaussian proposal)
                  for the Random Walk Metropolis kernel.

  """

  step_size: float = DEFAULT_STEP_SIZE
