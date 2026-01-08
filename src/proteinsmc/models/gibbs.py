"""Data structures for Gibbs sampler."""

from __future__ import annotations

from collections.abc import Callable

from jaxtyping import Array, PRNGKeyArray

from proteinsmc.models.fitness import FitnessFn
from proteinsmc.models.sampler_base import BaseSamplerConfig
from proteinsmc.models.types import EvoSequence

GibbsUpdateFn = Callable[
  [PRNGKeyArray | None, EvoSequence, FitnessFn, int, Array | None],
  EvoSequence,
]


class GibbsConfig(BaseSamplerConfig):
  """Configuration for the Gibbs sampler."""
