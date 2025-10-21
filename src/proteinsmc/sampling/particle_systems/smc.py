"""Core JIT-compiled logic for the SMC sampler."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from blackjax.smc import resampling
from blackjax.smc.base import SMCInfo
from blackjax.smc.base import SMCState as BlackjaxSMCState
from blackjax.smc.base import step as smc_step
from jax import jit
from jax.experimental import io_callback

if TYPE_CHECKING:
  from collections.abc import Callable

  from jaxtyping import Array, Float, PRNGKeyArray

  from proteinsmc.models.mutation import MutationFn

from proteinsmc.models.sampler_base import SamplerState
from proteinsmc.models.smc import (
  PopulationSequences,
  SMCAlgorithm,
)

if TYPE_CHECKING:
  from jaxtyping import Int, PRNGKeyArray

  from proteinsmc.models.annealing import AnnealingFn
  from proteinsmc.models.fitness import StackedFitnessFn


@dataclass
class SMCOutput:
  """Output of a single SMC step."""

  state: SamplerState
  info: SMCInfo


def resample(
  resampling_approach: str,
  key: PRNGKeyArray,
  weights: Float,
  num_samples: int,
) -> Array:
  """Resampling function based on the configured approach."""
  match resampling_approach:
    case "systematic":
      return resampling.systematic(key, weights, num_samples)
    case "multinomial":
      return resampling.multinomial(key, weights, num_samples)
    case "stratified":
      return resampling.stratified(key, weights, num_samples)
    case "residual":
      return resampling.residual(key, weights, num_samples)
    case _:
      msg = f"Unknown resampling approach: {resampling_approach}"
      raise ValueError(msg)


def create_smc_loop_func(
  algorithm: SMCAlgorithm,
  resampling_approach: str,
  weight_fn: StackedFitnessFn,
  mutation_fn: MutationFn,
) -> Callable[[BlackjaxSMCState, PRNGKeyArray], tuple[BlackjaxSMCState, SMCInfo]]:
  """Create a JIT-compiled SMC loop function."""
  match algorithm:
    case SMCAlgorithm.BASE | SMCAlgorithm.ANNEALING | SMCAlgorithm.PARALLEL_REPLICA:
      return partial(smc_step)(
        weight_fn=weight_fn,
        update_fn=mutation_fn,
        resample_fn=partial(resample, resampling_approach),
      )  # type: ignore[return-value]
    case SMCAlgorithm.ADAPTIVE_TEMPERED:
      msg = "Adaptive Tempered SMC algorithm is not implemented in the loop function."
      raise NotImplementedError(msg)
    case _:
      msg = f"SMC algorithm {algorithm} is not currently supported."
      raise NotImplementedError(msg)


@partial(jit, static_argnames=("fitness_fn", "mutation_fn", "annealing_fn"))
def run_smc_loop(  # noqa: PLR0913
  num_samples: int,
  algorithm: SMCAlgorithm,
  resampling_approach: str,
  initial_state: SamplerState,
  fitness_fn: StackedFitnessFn,
  mutation_fn: MutationFn,
  annealing_fn: AnnealingFn | None = None,
  writer_callback: Callable | None = None,
) -> tuple[SamplerState, SMCInfo]:
  """JIT-compiled SMC loop."""
  smc_loop_func = create_smc_loop_func(
    algorithm=algorithm,
    resampling_approach=resampling_approach,
    weight_fn=fitness_fn,
    mutation_fn=mutation_fn,
  )

  def scan_body(state: SamplerState, i: Int) -> tuple[SamplerState, SMCInfo]:
    current_beta = None if annealing_fn is None else annealing_fn(i, _context=None)  # type: ignore[call-arg]
    key_for_fitness_fn, key_for_blackjax, next_key = jax.random.split(
      state.key,
      3,
    )

    def weight_fn(
      sequence: PopulationSequences,
    ) -> Array:
      """Weight function for the SMC step."""
      return fitness_fn(key_for_fitness_fn, sequence, current_beta)

    next_state, info = smc_loop_func(
      state.blackjax_state,  # type: ignore[arg-type]
      key_for_blackjax,
    )
    next_smc_state = SamplerState(
      population=next_state.particles,  # type: ignore[call-arg]
      key=next_key,
      blackjax_state=next_state,
      step=i,
      additional_fields={
        "beta": current_beta if current_beta is not None else jnp.array(-1.0),
      },
    )
    if writer_callback is not None:
      smc_step_output = SMCOutput(
        state=next_smc_state,
        info=info,
      )
      io_callback(
        writer_callback,
        None,
        {"state": smc_step_output.state, "info": smc_step_output.info},
      )
    return next_smc_state, info

  final_state, collected_metrics = jax.lax.fori_loop(
    0,
    num_samples,
    lambda i, val: scan_body(val[0], i),
    (initial_state, None),
  )

  return final_state, collected_metrics
