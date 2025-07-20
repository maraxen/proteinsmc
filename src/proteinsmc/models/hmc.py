"""Data structures for HMC sampler."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING

from flax.struct import PyTreeNode

from proteinsmc.models.sampler_base import BaseSamplerConfig

if TYPE_CHECKING:
  from jaxtyping import Float, PRNGKeyArray

  from proteinsmc.models.types import EvoSequence


class HMCState(PyTreeNode):
  """State of the HMC sampler.

  Attributes:
      samples: An array of sampled sequences.
      fitness: The fitness of the last sampled sequence.

  """

  samples: EvoSequence
  fitness: Float
  key: PRNGKeyArray


class HMCConfig(BaseSamplerConfig):
  """Configuration for the HMC sampler.

  Attributes:
      num_samples: The number of samples to generate.
      step_size: The step size for the leapfrog integrator.
      num_leapfrog_steps: The number of leapfrog steps to take.

  """

  step_size: float = field(default=0.1)
  """Step size for the leapfrog integrator."""
  num_leapfrog_steps: int = field(default=10)
  """Number of leapfrog steps to take."""
