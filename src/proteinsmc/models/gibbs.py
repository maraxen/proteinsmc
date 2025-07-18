"""Data structures for Gibbs sampler."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, TypedDict, Unpack

from flax.struct import PyTreeNode

from proteinsmc.models.sampler_base import BaseSamplerConfig

if TYPE_CHECKING:
  from jaxtyping import Array, Float, PRNGKeyArray

  from proteinsmc.models.fitness import FitnessFuncSignature

from proteinsmc.models.types import EvoSequence


class GibbsUpdateKwargs(TypedDict):
  """TypedDict for the parameters of a Gibbs update function."""

  key: PRNGKeyArray
  sequence: EvoSequence
  fitness_fn: FitnessFuncSignature
  position: int

  _context: Array | None


GibbsUpdateFuncSignature = Callable[[Unpack[GibbsUpdateKwargs]], EvoSequence]


class GibbsState(PyTreeNode):
  """State of the Gibbs sampler.

  Attributes:
      samples: An array of sampled sequences.
      fitness: The fitness of the last sampled sequence.

  """

  samples: EvoSequence
  fitness: Float[Array, ""]
  key: PRNGKeyArray


class GibbsConfig(BaseSamplerConfig):
  """Configuration for the Gibbs sampler.

  Attributes:
      num_samples: The number of samples to generate.

  """

  num_samples: int
