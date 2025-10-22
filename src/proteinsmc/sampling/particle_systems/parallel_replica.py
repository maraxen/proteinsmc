"""Implementation of Parallel Replica inspired Sequential Monte Carlo (PRSMC) sampling."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from blackjax.smc.base import SMCInfo
from blackjax.smc.base import step as blackjax_smc_step
from flax import struct
from jax import lax, vmap
from jax.experimental import io_callback

from proteinsmc.utils.mutation import mutate

if TYPE_CHECKING:
  from collections.abc import Callable

  from jaxtyping import Array, Float, Int, PRNGKeyArray

  from proteinsmc.models.fitness import StackedFitnessFn
  from proteinsmc.models.types import SequenceType
  from proteinsmc.utils.annealing import AnnealingFn


from proteinsmc.models.sampler_base import SamplerState
from proteinsmc.sampling.particle_systems.smc import resample


@struct.dataclass
class MigrationInfo:
  """Information about replica exchange/migration events during a PRSMC step.

  All arrays have static shape to be JAX-compatible. The first dimension
  corresponds to the n_exchange_attempts (all attempted swaps, accepted or not).
  """

  island_from: Array  # Shape: (n_exchange_attempts,) - source island index
  island_to: Array  # Shape: (n_exchange_attempts,) - destination island index
  particle_idx_from: Array  # Shape: (n_exchange_attempts,) - source particle index
  particle_idx_to: Array  # Shape: (n_exchange_attempts,) - dest particle index
  accepted: Array  # Shape: (n_exchange_attempts,) - bool, was swap accepted
  log_acceptance_ratio: Array  # Shape: (n_exchange_attempts,) - acceptance probability


@struct.dataclass
class PRSMCOutput:
  """Output of a single PRSMC step, with data stacked across all islands.

  The `state` and `info` fields contain pytrees where the leading axis
  corresponds to the islands.
  """

  state: SamplerState
  info: SMCInfo
  num_attempted_swaps: Int
  num_accepted_swaps: Int
  migration_info: MigrationInfo  # Migration events for this step


def mutation_update_fn(
  keys: PRNGKeyArray,
  sequences: Array,
  update_parameters: dict[str, Array],
  q_states: Int,
) -> tuple[Array, None]:
  """Apply mutation to the resampled sequences."""
  mutated_sequence = jax.vmap(
    partial(mutate, q_states=q_states),
    in_axes=(0, 0, 0),
  )(keys, sequences, update_parameters["mutation_rate"])

  return mutated_sequence, None


def weight_fn(
  sequence: Array,
  fitness_fn_partial: Callable,
  beta: Float,
) -> Array:
  """Weight function for the SMC step for a single particle."""
  fitness_values, _ = fitness_fn_partial(sequence)
  # Squeeze to ensure we have a scalar fitness value
  fitness_scalar = jnp.squeeze(fitness_values)
  return jnp.array(
    jnp.where(
      jnp.isneginf(fitness_scalar),
      -jnp.inf,
      beta * fitness_scalar,
    ),
    dtype=jnp.float32,
  )


def migrate(  # noqa: PLR0913
  island_states: SamplerState,
  meta_beta: Float,
  key: PRNGKeyArray,
  n_islands: Int,
  population_size: Int,
  n_exchange_attempts: Int,
  fitness_fn: StackedFitnessFn,
) -> tuple[SamplerState, Int, MigrationInfo]:
  """Perform replica exchange attempts between islands.

  This function should only be called when n_islands > 1.

  Returns:
      Updated island states, number of accepted swaps, and migration info.

  """
  all_particles = island_states.blackjax_state.particles  # type: ignore[attr-access]
  all_betas = island_states.additional_fields["beta"]
  mean_fitness = island_states.additional_fields["mean_fitness"]

  num_accepted_swaps = jnp.array(0, dtype=jnp.int32)

  # Use mean_fitness (shape: n_islands) to determine selection probabilities
  safe_mean_fitness = jnp.nan_to_num(mean_fitness, nan=-jnp.inf)
  probs_idx1 = jax.nn.softmax(safe_mean_fitness)

  # Pre-allocate arrays for migration tracking (static shapes)
  island_from_log = jnp.zeros(n_exchange_attempts, dtype=jnp.int32)
  island_to_log = jnp.zeros(n_exchange_attempts, dtype=jnp.int32)
  particle_idx_from_log = jnp.zeros(n_exchange_attempts, dtype=jnp.int32)
  particle_idx_to_log = jnp.zeros(n_exchange_attempts, dtype=jnp.int32)
  accepted_log = jnp.zeros(n_exchange_attempts, dtype=jnp.bool_)
  log_acceptance_ratio_log = jnp.zeros(n_exchange_attempts, dtype=jnp.float32)

  def attempt_exchange_body(i: int, loop_state: tuple) -> tuple:
    (
      key_attempt,
      current_particles,
      current_accepted_swaps,
      islands_from,
      islands_to,
      particles_from,
      particles_to,
      accepted_arr,
      log_ratios,
    ) = loop_state
    (
      key_select_idx1,
      key_select_idx2,
      key_particle_choice,
      key_acceptance,
      key_next_iter,
    ) = jax.random.split(key_attempt, 5)

    idx1 = jax.random.choice(key_select_idx1, jnp.arange(n_islands), p=probs_idx1)
    idx2 = (idx1 + jax.random.randint(key_select_idx2, (), 1, n_islands)) % n_islands

    particle_idx = jax.random.randint(
      key_particle_choice,
      (2,),
      0,
      population_size,
    )
    particle1 = current_particles[idx1, particle_idx[0]]
    particle2 = current_particles[idx2, particle_idx[1]]

    fitness1, _ = fitness_fn(jnp.expand_dims(particle1, 0), key_acceptance, None)
    fitness2, _ = fitness_fn(jnp.expand_dims(particle2, 0), key_acceptance, None)

    log_acceptance_ratio = (
      meta_beta * (all_betas[idx1] - all_betas[idx2]) * (fitness2[0] - fitness1[0])
    )
    log_acceptance_ratio = jnp.array(
      jnp.where(
        jnp.isinf(fitness1[0]) | jnp.isinf(fitness2[0]),
        -jnp.inf,
        log_acceptance_ratio,
      ),
      dtype=jnp.float32,
    )

    accept = jnp.log(jax.random.uniform(key_acceptance)) < log_acceptance_ratio

    new_particles = lax.cond(
      accept,
      lambda p: p.at[idx1, particle_idx[0]]
      .set(particle2)
      .at[
        idx2,
        particle_idx[1],
      ]
      .set(particle1),
      lambda p: p,
      current_particles,
    )
    updated_accepted_swaps = current_accepted_swaps + jax.lax.select(accept, 1, 0)

    # Record migration event
    islands_from = islands_from.at[i].set(idx1)
    islands_to = islands_to.at[i].set(idx2)
    particles_from = particles_from.at[i].set(particle_idx[0])
    particles_to = particles_to.at[i].set(particle_idx[1])
    accepted_arr = accepted_arr.at[i].set(accept)
    log_ratios = log_ratios.at[i].set(log_acceptance_ratio)

    return (
      key_next_iter,
      new_particles,
      updated_accepted_swaps,
      islands_from,
      islands_to,
      particles_from,
      particles_to,
      accepted_arr,
      log_ratios,
    )

  (
    key,
    final_particles,
    total_accepted,
    island_from_log,
    island_to_log,
    particle_idx_from_log,
    particle_idx_to_log,
    accepted_log,
    log_acceptance_ratio_log,
  ) = lax.fori_loop(
    0,
    n_exchange_attempts,
    attempt_exchange_body,
    (
      key,
      all_particles,
      num_accepted_swaps,
      island_from_log,
      island_to_log,
      particle_idx_from_log,
      particle_idx_to_log,
      accepted_log,
      log_acceptance_ratio_log,
    ),
  )

  # Update blackjax_state.particles - create new blackjax state with updated particles
  updated_blackjax_state = island_states.blackjax_state._replace(particles=final_particles)  # type: ignore[union-attr]

  # Create new SamplerState with updated blackjax_state using .replace()
  updated_states = island_states.replace(blackjax_state=updated_blackjax_state)  # type: ignore[call-arg]

  migration_info = MigrationInfo(
    island_from=island_from_log,
    island_to=island_to_log,
    particle_idx_from=particle_idx_from_log,
    particle_idx_to=particle_idx_to_log,
    accepted=accepted_log,
    log_acceptance_ratio=log_acceptance_ratio_log,
  )

  return updated_states, total_accepted, migration_info


def run_prsmc_loop(  # noqa: PLR0913
  num_steps: int,
  initial_state: SamplerState,
  sequence_type: SequenceType,
  resampling_approach: str,
  population_size: Int,
  n_islands: Int,
  exchange_frequency: Int,
  n_exchange_attempts: Int,
  fitness_fn: StackedFitnessFn,
  annealing_fn: AnnealingFn,
  writer_callback: Callable,
) -> SamplerState:
  """JIT-compiled Parallel Replica SMC loop."""
  q_states = 4 if sequence_type == "nucleotide" else 20
  mutation_partial = partial(
    mutation_update_fn,
    q_states=q_states,
  )

  def _prsmc_step(
    carry_state: SamplerState,
    step_idx: Int,
  ) -> tuple[SamplerState, None]:
    # carry_state.key has shape (n_islands, 2) - split per island
    # vmap over islands to split keys: (n_islands, 2) -> 4 x (n_islands, 2)
    split_keys = vmap(lambda k: jax.random.split(k, 4))(carry_state.key)
    keys_islands = split_keys[:, 0, :]  # (n_islands, 2) - for SMC step
    keys_fitness_islands = split_keys[:, 1, :]  # (n_islands, 2) - for fitness eval
    key_exchange = split_keys[0, 2, :]  # (2,) - single key for exchange (scalar across islands)
    key_next_loop = split_keys[:, 3, :]  # (n_islands, 2) - for next iteration
    betas = carry_state.additional_fields["beta"]

    def single_island_weight_fn(beta: Float, key: PRNGKeyArray, sequence: Array) -> Array:
      fitness_fn_partial = partial(fitness_fn, key, None)
      return weight_fn(sequence, fitness_fn_partial, beta)

    # Batch the weight function for all particles
    def batched_island_weight_fn(beta: Float, key: PRNGKeyArray, sequences: Array) -> Array:
      """Weight function that takes all particles and returns all weights."""
      return vmap(lambda seq: single_island_weight_fn(beta, key, seq))(sequences)

    smc_step_fn = partial(
      blackjax_smc_step,
      update_fn=mutation_partial,
      resample_fn=partial(resample, resampling_approach),
    )

    # Apply one SMC step to each island
    next_blackjax_states, infos = vmap(
      lambda state, key, beta: smc_step_fn(
        rng_key=key,
        state=state,
        weight_fn=partial(batched_island_weight_fn, beta, key),
      ),
      in_axes=(0, 0, 0),
    )(carry_state.blackjax_state, keys_islands, betas)

    def extract_fitness_for_island(particles: Array, key: PRNGKeyArray) -> Array:
      """Extract fitness values for all particles in one island."""
      fitness_vec = vmap(lambda seq: fitness_fn(seq, key, None))(particles)

      return fitness_vec[0]

    fitness_values = vmap(extract_fitness_for_island, in_axes=(0, 0))(
      next_blackjax_states.particles,  # type: ignore[attr-access]
      keys_fitness_islands,
    )

    fitness_for_metrics = (
      jnp.squeeze(fitness_values, axis=-1)
      if fitness_values.ndim > 2  # noqa: PLR2004
      else fitness_values
    )
    ess = 1.0 / jnp.sum(next_blackjax_states.weights**2, axis=-1, keepdims=False)
    mean_fitness = jnp.nansum(
      fitness_for_metrics * next_blackjax_states.weights,
      axis=-1,
      keepdims=False,
    )
    max_fitness = jnp.max(
      jnp.array(jnp.where(jnp.isfinite(fitness_for_metrics), fitness_for_metrics, -jnp.inf)),
      axis=-1,
      keepdims=False,
    )

    logZ_estimates = carry_state.additional_fields["logZ_estimate"] + infos.log_likelihood_increment  # noqa: N806

    state_after_smc = SamplerState(
      sequence=next_blackjax_states.particles,  # type: ignore[call-arg]
      fitness=fitness_values,
      key=key_next_loop,
      blackjax_state=next_blackjax_states,
      step=carry_state.step + 1,
      additional_fields={
        "beta": betas,
        "mean_fitness": mean_fitness,
        "max_fitness": max_fitness,
        "ess": ess,
        "logZ_estimate": logZ_estimates,
      },
    )

    meta_beta = annealing_fn(step_idx)
    do_exchange = (step_idx + 1) % exchange_frequency == 0

    empty_migration_info = MigrationInfo(
      island_from=jnp.zeros(n_exchange_attempts, dtype=jnp.int32),
      island_to=jnp.zeros(n_exchange_attempts, dtype=jnp.int32),
      particle_idx_from=jnp.zeros(n_exchange_attempts, dtype=jnp.int32),
      particle_idx_to=jnp.zeros(n_exchange_attempts, dtype=jnp.int32),
      accepted=jnp.zeros(n_exchange_attempts, dtype=jnp.bool_),
      log_acceptance_ratio=jnp.zeros(n_exchange_attempts, dtype=jnp.float32),
    )

    state_after_migration, num_accepted_swaps, migration_info = lax.cond(
      do_exchange & (n_islands > 1),
      lambda: migrate(
        state_after_smc,
        meta_beta,
        key_exchange,
        n_islands,
        population_size,
        n_exchange_attempts,
        fitness_fn,
      ),
      lambda: (state_after_smc, jnp.array(0, dtype=jnp.int32), empty_migration_info),
    )
    num_attempted_swaps = jnp.where(
      do_exchange & (n_islands > 1),
      n_exchange_attempts,
      0,
    )

    step_output = PRSMCOutput(
      state=state_after_migration,
      info=infos,
      num_attempted_swaps=num_attempted_swaps,
      num_accepted_swaps=num_accepted_swaps,
      migration_info=migration_info,
    )
    io_callback(
      writer_callback,
      None,
      {
        "state": step_output.state,
        "info": step_output.info,
        "num_attempted_swaps": step_output.num_attempted_swaps,
        "num_accepted_swaps": step_output.num_accepted_swaps,
        "migration_info": step_output.migration_info,
      },
    )

    return state_after_migration, None

  current_state = initial_state

  def loop_body(i: int, state: SamplerState) -> SamplerState:
    """Loop body for fori_loop."""
    next_state, _ = _prsmc_step(state, i)
    return next_state

  return lax.fori_loop(
    0,
    num_steps,
    loop_body,
    current_state,
  )
