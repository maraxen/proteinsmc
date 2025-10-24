"""Implementation of Parallel Replica inspired Sequential Monte Carlo (PRSMC) sampling."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from blackjax.smc.base import step as blackjax_smc_step
from jax import lax, vmap
from jax.experimental import io_callback

if TYPE_CHECKING:
  from collections.abc import Callable

  from jaxtyping import Array, Float, Int, PRNGKeyArray

  from proteinsmc.models.fitness import StackedFitnessFn
  from proteinsmc.models.mutation import MutationFn
  from proteinsmc.utils.annealing import AnnealingFn


from proteinsmc.models.sampler_base import SamplerOutput, SamplerState
from proteinsmc.sampling.particle_systems.smc import resample


def migrate(  # noqa: PLR0913
  island_states: SamplerState,
  meta_beta: Float,
  key: PRNGKeyArray,
  n_islands: Int,
  population_size: Int,
  n_exchange_attempts: Int,
  fitness_fn: StackedFitnessFn,
) -> tuple[
  SamplerState,
  Int,
  Array,
  Array,
  Array,
  Array,
  Array,
  Array,
]:
  """Perform replica exchange attempts between islands.

  This function should only be called when n_islands > 1.

  Returns:
      Tuple containing:
      - Updated island states
      - Number of accepted swaps
      - island_from array
      - island_to array
      - particle_idx_from array
      - particle_idx_to array
      - accepted array
      - log_acceptance_ratio array

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

    fitness1 = fitness_fn(key_acceptance, jnp.expand_dims(particle1, 0), all_betas[idx1])
    fitness2 = fitness_fn(key_acceptance, jnp.expand_dims(particle2, 0), all_betas[idx2])

    log_acceptance_ratio = (
      meta_beta
      * (all_betas[idx1] - all_betas[idx2])
      * (jnp.squeeze(fitness2) - jnp.squeeze(fitness1))
    )
    log_acceptance_ratio = jnp.array(
      jnp.where(
        jnp.isinf(fitness1) | jnp.isinf(fitness2),
        -jnp.inf,
        log_acceptance_ratio,
      ),
      dtype=jnp.float32,
    )
    log_acceptance_ratio = jnp.squeeze(log_acceptance_ratio)

    accept = jnp.log(jax.random.uniform(key_acceptance)) < log_acceptance_ratio
    accept = jnp.squeeze(accept)

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

  return (
    updated_states,
    total_accepted,
    island_from_log,
    island_to_log,
    particle_idx_from_log,
    particle_idx_to_log,
    accepted_log,
    log_acceptance_ratio_log,
  )


@partial(
    jax.jit,
    static_argnames=(
        "chunk_size",
        "resampling_approach",
        "population_size",
        "n_islands",
        "exchange_frequency",
        "n_exchange_attempts",
        "fitness_fn",
        "mutation_fn",
        "annealing_fn",
    ),
)
def _run_prsmc_chunk(
    chunk_size: int,
    initial_state: SamplerState,
    resampling_approach: str,
    population_size: Int,
    n_islands: Int,
    exchange_frequency: Int,
    n_exchange_attempts: Int,
    fitness_fn: StackedFitnessFn,
    mutation_fn: MutationFn,
    annealing_fn: AnnealingFn,
) -> tuple[SamplerState, SamplerOutput]:
    """JIT-compiled Parallel Replica SMC loop for a single chunk."""
    vmap_mutation_fn = jax.vmap(mutation_fn, in_axes=(0, 0, None))

    def _prsmc_step(
        carry_state: SamplerState, _: Int
    ) -> tuple[SamplerState, SamplerOutput]:
        split_keys = vmap(lambda k: jax.random.split(k, 4))(carry_state.key)
        keys_islands = split_keys[:, 0, :]
        _ = split_keys[:, 1, :]
        key_exchange = split_keys[0, 2, :]
        key_next_loop = split_keys[:, 3, :]
        betas = carry_state.additional_fields["beta"]

        def single_island_weight_fn(
            beta: Float, key: PRNGKeyArray, sequence: Array
        ) -> Array:
            return fitness_fn(key, sequence, beta)

        def batched_island_weight_fn(
            beta: Float, key: PRNGKeyArray, sequences: Array
        ) -> Array:
            fitness_keys = jax.random.split(key, sequences.shape[0])
            return vmap(single_island_weight_fn, in_axes=(None, 0, 0))(
                beta, fitness_keys, sequences
            )

        smc_step_fn = partial(
            blackjax_smc_step,
            update_fn=vmap_mutation_fn,
            resample_fn=partial(resample, resampling_approach),
        )

        next_blackjax_states, infos = vmap(
            lambda state, key, beta: smc_step_fn(
                rng_key=key,
                state=state,
                weight_fn=partial(batched_island_weight_fn, beta, key),
            ),
            in_axes=(0, 0, 0),
        )(carry_state.blackjax_state, keys_islands, betas)

        ess = 1.0 / jnp.sum(next_blackjax_states.weights ** 2, axis=-1, keepdims=False)
        mean_fitness = jnp.nansum(
            next_blackjax_states.weights,
            axis=-1,
            keepdims=False,
        )
        max_fitness = jnp.max(
            jnp.array(
                jnp.where(
                    jnp.isfinite(next_blackjax_states.weights),
                    next_blackjax_states.weights,
                    -jnp.inf,
                )
            ),
            axis=-1,
            keepdims=False,
        )

        logZ_estimates = (
            carry_state.additional_fields["logZ_estimate"]
            + infos.log_likelihood_increment
        )

        state_after_smc = SamplerState(
            sequence=next_blackjax_states.particles,
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

        meta_beta = annealing_fn(current_step=carry_state.step, _context=None)
        do_exchange = (carry_state.step + 1) % exchange_frequency == 0

        empty_island_from = jnp.zeros(n_exchange_attempts, dtype=jnp.int32)
        empty_island_to = jnp.zeros(n_exchange_attempts, dtype=jnp.int32)
        empty_particle_from = jnp.zeros(n_exchange_attempts, dtype=jnp.int32)
        empty_particle_to = jnp.zeros(n_exchange_attempts, dtype=jnp.int32)
        empty_accepted = jnp.zeros(n_exchange_attempts, dtype=jnp.bool_)
        empty_log_ratio = jnp.zeros(n_exchange_attempts, dtype=jnp.float32)

        (
            state_after_migration,
            num_accepted_swaps,
            island_from,
            island_to,
            particle_from,
            particle_to,
            accepted,
            log_ratio,
        ) = lax.cond(
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
            lambda: (
                state_after_smc,
                jnp.array(0, dtype=jnp.int32),
                empty_island_from,
                empty_island_to,
                empty_particle_from,
                empty_particle_to,
                empty_accepted,
                empty_log_ratio,
            ),
        )
        num_attempted_swaps = jnp.array(
            jnp.where(
                do_exchange & (n_islands > 1),
                n_exchange_attempts,
                0,
            ),
            dtype=jnp.int32,
        )

        step_output = SamplerOutput(
            step=state_after_migration.step,
            sequences=state_after_migration.sequence,
            key=key_next_loop[0],
            fitness=next_blackjax_states.weights,
            weights=next_blackjax_states.weights,
            log_likelihood_increment=jnp.atleast_1d(infos.log_likelihood_increment)[
                0
            ],
            ancestors=infos.ancestors,
            ess=ess,
            beta=betas[0],
            num_attempted_swaps=num_attempted_swaps,
            num_accepted_swaps=num_accepted_swaps,
            migration_island_from=island_from,
            migration_island_to=island_to,
            migration_particle_idx_from=particle_from,
            migration_particle_idx_to=particle_to,
            migration_accepted=accepted,
            migration_log_acceptance_ratio=log_ratio,
            mean_fitness=mean_fitness[0],
            max_fitness=max_fitness[0],
            log_z_estimate=logZ_estimates[0],
        )

        return state_after_migration, step_output

    final_state, collected_outputs = lax.scan(
        _prsmc_step, initial_state, jnp.arange(chunk_size)
    )
    return final_state, collected_outputs


def run_prsmc_loop(  # noqa: PLR0913
    num_steps: int,
    initial_state: SamplerState,
    resampling_approach: str,
    population_size: Int,
    n_islands: Int,
    exchange_frequency: Int,
    n_exchange_attempts: Int,
    fitness_fn: StackedFitnessFn,
    mutation_fn: MutationFn,
    annealing_fn: AnnealingFn,
    writer_callback: Callable,
    chunk_size: int = 100,
) -> SamplerState:
    """Orchestrator for the Parallel Replica SMC loop, running in chunks."""
    num_chunks = (num_steps + chunk_size - 1) // chunk_size
    current_state = initial_state

    for i in range(num_chunks):
        current_chunk_size = min(chunk_size, num_steps - i * chunk_size)
        final_state, collected_outputs = _run_prsmc_chunk(
            chunk_size=current_chunk_size,
            initial_state=current_state,
            resampling_approach=resampling_approach,
            population_size=population_size,
            n_islands=n_islands,
            exchange_frequency=exchange_frequency,
            n_exchange_attempts=n_exchange_attempts,
            fitness_fn=fitness_fn,
            mutation_fn=mutation_fn,
            annealing_fn=annealing_fn,
        )
        writer_callback(collected_outputs)
        current_state = final_state

    return current_state
