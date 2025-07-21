"""Data structures for Gibbs sampler."""

from __future__ import annotations

from typing import Callable

from flax.struct import PyTreeNode
from jaxtyping import Array, Float, PRNGKeyArray

from proteinsmc.models.fitness import FitnessFn
from proteinsmc.models.sampler_base import BaseSamplerConfig
from proteinsmc.models.types import EvoSequence

GibbsUpdateFn = Callable[
  [EvoSequence, PRNGKeyArray | None, FitnessFn, int, Array | None],
  EvoSequence,
]


class GibbsState(PyTreeNode):
  """State of the Gibbs sampler.

  Attributes:
      sequences: An array of sampled sequences.
      fitness: The fitness of the last sampled sequence.
      key: The JAX PRNG key for the next step.

  """

  sequences: EvoSequence
  fitness: Float
  key: PRNGKeyArray


class GibbsConfig(BaseSamplerConfig):
  """Configuration for the Gibbs sampler."""
