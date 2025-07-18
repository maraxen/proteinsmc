"""Data structures for NUTS sampler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flax.struct import PyTreeNode

from proteinsmc.models.sampler_base import BaseSamplerConfig

if TYPE_CHECKING:
  from jaxtyping import Array, Float, PRNGKeyArray

  from proteinsmc.models.types import EvoSequence


class NUTSState(PyTreeNode):
  """State of the NUTS sampler.

  Attributes:
      samples: An array of sampled sequences.
      fitness: The fitness of the last sampled sequence.

  """

  samples: EvoSequence
  fitness: Float[Array, ""]
  key: PRNGKeyArray


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

  num_samples: int
  step_size: float
  num_leapfrog_steps: int
  warmup_steps: int
  num_chains: int
  adapt_step_size: bool = True
