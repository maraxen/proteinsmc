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


# register SMCInfo as a pytree node
def _smcinfo_flatten(smc_info: SMCInfo):
  return (smc_info.ancestors, smc_info.log_likelihood_increment, smc_info.update_info), None


def _smcinfo_unflatten(aux_data, children):
  ancestors, lli, update_info = children
  return SMCInfo(ancestors, lli, update_info)


jax.tree_util.register_pytree_node(
  SMCInfo,
  _smcinfo_flatten,
  _smcinfo_unflatten,
)


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
  weight_fn: StackedFitnessFn,
  mutation_fn: MutationFn,
) -> Callable[[BlackjaxSMCState, PRNGKeyArray], tuple[BlackjaxSMCState, SMCInfo]]:
  """Create a JIT-compiled SMC loop function."""
  match algorithm:
    case "BaseSMC" | "AnnealedSMC" | "ParallelReplicaSMC":
      return partial(
        smc_step,
        weight_fn=weight_fn,
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
  annealing_fn: AnnealingFn | None = None,
  writer_callback: Callable | None = None,
) -> tuple[SamplerState, SMCInfo]:
  """JIT-compiled SMC loop."""
  # TODO: Get mutation_fn and fitness_fn loaded as vmapped transformation, we should standardize this across the codebase, following whatever is clearest, best for maintainability/extensibility, and follows best practices and handle keys
  # because handling keys is crucial for reproducibility and debugging, we should standardize it across the codebase
  # mutation_fn: (sequences, keys, context) -> new_sequences
  # fitness_fn: (keys, sequences, context) -> weights, but blackjax expects ONLY particles to be fed into the weights fn
  vmappped_mutation_fn = jax.vmap(mutation_fn, in_axes=(0, 0, None))
  fitness_fn = partial(fitness_fn, key=jax.random.PRNGKey(0))

  def weight_fn(sequence: PopulationSequences) -> Array:
    """Weight function for the SMC step."""
    return fitness_fn(sequence)[0]

  smc_loop_func = create_smc_loop_func(
    algorithm=algorithm,
    resampling_approach=resampling_approach,
    weight_fn=weight_fn,
    mutation_fn=vmappped_mutation_fn,
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
      key_for_blackjax,
      state.blackjax_state,  # type: ignore[arg-type]
    )
    next_smc_state = SamplerState(
      sequence=next_state.particles,  # type: ignore[call-arg]
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
        smc_step_output,
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
        update_info={},
      ),
    ),
  )

  return final_state, collected_metrics
