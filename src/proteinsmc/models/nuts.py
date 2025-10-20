"""Data structures for NUTS sampler."""

from __future__ import annotations

from dataclasses import dataclass, field

from proteinsmc.models.sampler_base import BaseSamplerConfig


@dataclass(frozen=True)
class NUTSConfig(BaseSamplerConfig):
  """Configuration for the NUTS sampler.

  Attributes:
      num_samples: The number of samples to generate.
      step_size: The step size for the leapfrog integrator.
      num_leapfrog_steps: The number of leapfrog steps to take.
      warmup_steps: Number of warmup steps to adapt the step size.
      num_chains: Number of parallel chains to run.
      adapt_step_size: Whether to adapt the step size during warmup.

  """

  step_size: float = field(default=0.1)
  max_num_doublings: int = field(default=10)
