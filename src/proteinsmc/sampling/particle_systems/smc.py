"""Core JIT-compiled logic for the SMC sampler."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from blackjax.smc import resampling
from blackjax.smc.base import SMCInfo
from blackjax.smc.base import SMCState as BlackjaxSMCState
from blackjax.smc.base import step as smc_step
from flax.struct import dataclass
from jax.experimental import io_callback

if TYPE_CHECKING:
  from collections.abc import Callable

  from jaxtyping import Array, Float, PRNGKeyArray

  from proteinsmc.models.mutation import MutationFn

from proteinsmc.models.sampler_base import SamplerState

if TYPE_CHECKING:
  from jaxtyping import Int, PRNGKeyArray

  from proteinsmc.models.annealing import AnnealingFn
  from proteinsmc.models.fitness import StackedFitnessFn
  from proteinsmc.models.smc import PopulationSequences, SMCAlgorithmType


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
  algorithm: SMCAlgorithmType,
  resampling_approach: str,
  mutation_fn: MutationFn,
) -> Callable[[BlackjaxSMCState, PRNGKeyArray], tuple[BlackjaxSMCState, SMCInfo]]:
  """Create a JIT-compiled SMC loop function."""
  match algorithm:
    case "BaseSMC" | "AnnealedSMC" | "ParallelReplicaSMC":
      return partial(
        smc_step,
        update_fn=mutation_fn,
        resample_fn=partial(resample, resampling_approach),
      )  # type: ignore[return-value]
    case "AdaptiveTemperedSMC":
      msg = "Adaptive Tempered SMC algorithm is not implemented in the loop function."
      raise NotImplementedError(msg)
    case _:
      msg = f"SMC algorithm {algorithm} is not currently supported."
      raise NotImplementedError(msg)


class UpdateInfo(NamedTuple):
  """Placeholder for update info in SMCInfo."""


def run_smc_loop(  # noqa: PLR0913
  num_samples: int,
  algorithm: SMCAlgorithmType,
  resampling_approach: str,
  initial_state: SamplerState,
  fitness_fn: StackedFitnessFn,
  mutation_fn: MutationFn,
  writer_callback: Callable,
  annealing_fn: AnnealingFn | None = None,
) -> tuple[SamplerState, SMCInfo]:
  """JIT-compiled SMC loop."""
  # TODO: Get mutation_fn and fitness_fn loaded as vmapped transformation, we should standardize this across the codebase, following whatever is clearest, best for maintainability/extensibility, and follows best practices and handle keys
  # because handling keys is crucial for reproducibility and debugging, we should standardize it across the codebase
  # mutation_fn: (sequences, keys, context) -> new_sequences
  # fitness_fn: (keys, sequences, context) -> weights, but blackjax expects ONLY particles to be fed into the weights fn
  vmap_mutation_fn = jax.vmap(mutation_fn, in_axes=(0, 0, None))

  smc_loop_func_partial = create_smc_loop_func(
    algorithm=algorithm,
    resampling_approach=resampling_approach,
    mutation_fn=vmap_mutation_fn,
  )

  def scan_body(state: SamplerState, i: Int) -> tuple[SamplerState, SMCInfo]:
    current_beta = None if annealing_fn is None else annealing_fn(i, _context=None)  # type: ignore[call-arg]
    key_for_fitness_fn, key_for_blackjax, next_key = jax.random.split(
      state.key,
      3,
    )
    fitness_keys = jax.random.split(key_for_fitness_fn, state.sequence.shape[0])

    def weight_fn(
      sequence: PopulationSequences,
    ) -> Array:
      """Weight function for the SMC step."""
      return jax.vmap(fitness_fn, in_axes=(0, 0, None))(fitness_keys, sequence, current_beta)

    smc_loop_func_partial_with_weights = partial(
      smc_loop_func_partial,
      weight_fn=weight_fn,  # type: ignore[arg-type]
    )

    next_state, info = smc_loop_func_partial_with_weights(
      key_for_blackjax,
      state.blackjax_state,  # type: ignore[arg-type]
    )
    next_smc_state = SamplerState(
      sequence=next_state.particles,  # type: ignore[call-arg]
      key=next_key,
      blackjax_state=next_state,
      step=i + 1,
      additional_fields={
        "beta": current_beta if current_beta is not None else jnp.array(-1.0),
      },
    )
    io_callback(
      writer_callback,
      None,
      SMCOutput(
        state=next_smc_state,
        info=info,
      ),
    )
    return next_smc_state, info

  final_state, collected_metrics = jax.lax.fori_loop(
    0,
    num_samples,
    lambda i, val: scan_body(val[0], i),
    (
      initial_state,
      SMCInfo(
        ancestors=jnp.zeros((initial_state.sequence.shape[0],), dtype=jnp.int32),
        log_likelihood_increment=0.0,
        update_info=UpdateInfo(),
      ),
    ),
  )

  return final_state, collected_metrics
