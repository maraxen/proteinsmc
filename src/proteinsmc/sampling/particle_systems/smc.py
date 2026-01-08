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

from proteinsmc.models.sampler_base import SamplerOutput, SamplerState

if TYPE_CHECKING:
  from collections.abc import Callable

  from jaxtyping import Array, Float, Int, PRNGKeyArray

  from proteinsmc.models.protocols import AnnealingFn, FitnessFn, MutationFn
  from proteinsmc.models.smc import PopulationSequences, SMCAlgorithmType


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
) -> Callable[[PRNGKeyArray, BlackjaxSMCState], tuple[BlackjaxSMCState, SMCInfo]]:
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


if TYPE_CHECKING:

  def _run_smc_chunk(  # noqa: PLR0913
    chunk_size: int,
    algorithm: SMCAlgorithmType,
    resampling_approach: str,
    initial_state: SamplerState,
    fitness_fn: FitnessFn,
    mutation_fn: MutationFn,
    annealing_fn: AnnealingFn | None = None,
  ) -> tuple[SamplerState, SamplerOutput]: ...
else:

  @partial(
    jax.jit,
    static_argnames=(
      "chunk_size",
      "algorithm",
      "resampling_approach",
      "fitness_fn",
      "mutation_fn",
      "annealing_fn",
    ),
  )
  def _run_smc_chunk(  # noqa: PLR0913
    chunk_size: int,
    algorithm: SMCAlgorithmType,
    resampling_approach: str,
    initial_state: SamplerState,
    fitness_fn: FitnessFn,
    mutation_fn: MutationFn,
    annealing_fn: AnnealingFn | None = None,
  ) -> tuple[SamplerState, SamplerOutput]:
    """JIT-compiled SMC loop for a single chunk."""
    vmap_mutation_fn = jax.vmap(mutation_fn, in_axes=(0, 0, None))
    smc_loop_func_partial = create_smc_loop_func(
      algorithm=algorithm,
      resampling_approach=resampling_approach,
      mutation_fn=vmap_mutation_fn,
    )

    def scan_body(state: SamplerState, _i: Int) -> tuple[SamplerState, SamplerOutput]:
      current_beta = None if annealing_fn is None else annealing_fn(state.step)
      key_for_fitness_fn, key_for_blackjax, next_key = jax.random.split(state.key, 3)
      fitness_keys = jax.random.split(key_for_fitness_fn, state.sequence.shape[0])

      def weight_fn(sequence: PopulationSequences) -> Array:
        # fitness_fn returns stacked fitness: [combined, score1, score2, ...]
        # We extract only the combined fitness (first element) for SMC weights
        stacked_fitness = jax.vmap(fitness_fn, in_axes=(0, 0, None))(
          fitness_keys, sequence, current_beta
        )
        return stacked_fitness[:, 0]  # Extract combined fitness for each particle

      smc_loop_func_partial_with_weights = partial(
        smc_loop_func_partial,
        weight_fn=weight_fn,  # pyright: ignore[reportCallIssue]
      )
      next_blackjax_state, info = smc_loop_func_partial_with_weights(
        key_for_blackjax,
        state.blackjax_state,  # pyright: ignore[reportArgumentType]
      )
      next_smc_state = SamplerState(
        sequence=jnp.array(next_blackjax_state.particles),
        key=next_key,
        blackjax_state=next_blackjax_state,
        step=state.step + 1,
        additional_fields={
          "beta": current_beta if current_beta is not None else jnp.array(-1.0),
        },
      )
      sampler_output = SamplerOutput(
        step=next_smc_state.step,
        sequences=jnp.array(next_blackjax_state.particles),
        fitness=next_blackjax_state.weights,
        key=next_key,
        weights=next_blackjax_state.weights,
        log_likelihood_increment=jnp.array(info.log_likelihood_increment),
        ancestors=info.ancestors,
        ess=jnp.array(1.0 / jnp.sum(next_blackjax_state.weights**2)),
        beta=current_beta if current_beta is not None else jnp.array(-1.0),
      )
      return next_smc_state, sampler_output

    final_state, collected_outputs = jax.lax.scan(
      scan_body,
      initial_state,
      jnp.arange(chunk_size),
    )
    return final_state, collected_outputs


def run_smc_loop(  # noqa: PLR0913
  num_samples: int,
  algorithm: SMCAlgorithmType,
  resampling_approach: str,
  initial_state: SamplerState,
  fitness_fn: FitnessFn,
  mutation_fn: MutationFn,
  writer_callback: Callable,
  annealing_fn: AnnealingFn | None = None,
  chunk_size: int = 100,
) -> SamplerState:
  """Orchestrator for the SMC loop, running in chunks."""
  num_chunks = (num_samples + chunk_size - 1) // chunk_size
  current_state = initial_state

  for i in range(num_chunks):
    current_chunk_size = min(chunk_size, num_samples - i * chunk_size)
    final_state, collected_outputs = _run_smc_chunk(
      chunk_size=current_chunk_size,
      algorithm=algorithm,
      resampling_approach=resampling_approach,
      initial_state=current_state,
      fitness_fn=fitness_fn,
      mutation_fn=mutation_fn,
      annealing_fn=annealing_fn,
    )
    writer_callback(collected_outputs)
    current_state = final_state

  return current_state
