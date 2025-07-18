"""Data structures for MCMC sampler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flax.struct import PyTreeNode

from proteinsmc.models.sampler_base import BaseSamplerConfig

if TYPE_CHECKING:
  from jaxtyping import Array, Float, PRNGKeyArray

  from proteinsmc.models.types import EvoSequence


class MCMCState(PyTreeNode):
  """State of the MCMC sampler.

  Attributes:
      samples: An array of sampled sequences.
      fitness: The fitness of the last sampled sequence.

  """

  samples: EvoSequence
  fitness: Float[Array, ""]
  key: PRNGKeyArray


class MCMCConfig(BaseSamplerConfig):
  """Configuration for the MCMC sampler.

  Attributes:
      num_samples: The number of samples to generate.

  """

  num_samples: int
