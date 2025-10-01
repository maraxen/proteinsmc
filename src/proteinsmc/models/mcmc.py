"""Data structures for MCMC sampler."""

from __future__ import annotations

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

from flax.struct import PyTreeNode

from proteinsmc.models.sampler_base import BaseSamplerConfig

if TYPE_CHECKING:
  from blackjax.mcmc.random_walk import RWState as BlackjaxState
  from jaxtyping import PRNGKeyArray

  from proteinsmc.models.fitness import StackedFitness
  from proteinsmc.models.types import EvoSequence

DEFAULT_STEP_SIZE = 1e-1


@dataclass(frozen=True)
class MCMCConfig(BaseSamplerConfig):
  """Configuration for the MCMC sampler using a Random Walk Metropolis kernel.

  Attributes:
      step_size: The step size (standard deviation of the Gaussian proposal)
                  for the Random Walk Metropolis kernel.

  """

  step_size: float = DEFAULT_STEP_SIZE


class MCMCState(PyTreeNode):
  """State of the MCMC sampler.

  Attributes:
      sequence: The current sequence (sequence) of the sampler.
      fitness: The log-probability (fitness) of the current sequence.
      components_fitness: The individual components of the fitness function.
      key: The JAX PRNG key for the next step.
      blackjax_state: The internal state of the Blackjax sampler.

  """

  sequence: EvoSequence
  fitness: StackedFitness
  key: PRNGKeyArray
  blackjax_state: BlackjaxState
