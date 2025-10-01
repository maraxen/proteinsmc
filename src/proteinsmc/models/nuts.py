"""Data structures for NUTS sampler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from flax.struct import PyTreeNode

from proteinsmc.models.sampler_base import BaseSamplerConfig

if TYPE_CHECKING:
  from blackjax.base import State as BlackjaxState
  from jaxtyping import PRNGKeyArray

  from proteinsmc.models.fitness import StackedFitness
  from proteinsmc.models.types import EvoSequence


class NUTSState(PyTreeNode):
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
