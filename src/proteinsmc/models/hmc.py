"""Data structures for HMC sampler."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING

from flax.struct import PyTreeNode

from proteinsmc.models.sampler_base import BaseSamplerConfig

if TYPE_CHECKING:
  from blackjax.base import State as BlackjaxState
  from jaxtyping import PRNGKeyArray

  from proteinsmc.models.fitness import StackedFitness
  from proteinsmc.models.types import EvoSequence


class HMCState(PyTreeNode):
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
