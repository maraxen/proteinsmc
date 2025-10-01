"""Implementation of Parallel Replica inspired Sequential Monte Carlo (PRSMC) sampling."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
from blackjax.smc.base import SMCState as BlackjaxSMCState
from blackjax.smc.base import step as blackjax_smc_step
from jax import jit, lax, vmap

if TYPE_CHECKING:
  from jaxtyping import Array, Float, Int, PRNGKeyArray

  from proteinsmc.models.fitness import StackedFitnessFn
  from proteinsmc.utils.annealing import AnnealingFn
from proteinsmc.models.parallel_replica import (
  ExchangeConfig,
  IslandState,
  ParallelReplicaConfig,
  PRSMCState,
)
from proteinsmc.sampling.particle_systems.smc import resample
from proteinsmc.utils import (
  diversify_initial_sequences,
)
from proteinsmc.utils.initiate import generate_template_population
from proteinsmc.utils.mutation import mutate


def initialize_parallel_replica_state(
  config: ParallelReplicaConfig,
  _fitness_function: StackedFitnessFn,
  key: PRNGKeyArray,
) -> PRSMCState:
  """Initialize the state for a parallel replica in a JAX-idiomatic way."""
  total_pop_size = config.smc_config.population_size * config.n_islands
  template_population = generate_template_population(
    config.seed_sequence,
    total_pop_size,
    config.sequence_type,
    config.sequence_type,
  )
  template_population = diversify_initial_sequences(
    key=key,
    seed_sequences=template_population,
    mutation_rate=config.diversification_ratio,
    sequence_type=config.sequence_type,
  )
  template_population_stacked = template_population.reshape(
    config.n_islands,
    config.smc_config.population_size,
    -1,
  )

  island_keys = jax.random.split(key, config.n_islands)

  if config.track_lineage:
    island_pop_size = config.smc_config.population_size
    island_indices = jnp.arange(config.n_islands)[:, None]
    particle_indices = jnp.arange(island_pop_size)[None, :]
    global_ids = island_indices * island_pop_size + particle_indices
    parent_ids = jnp.full_like(global_ids, -1)  # -1 indicates the root

    lineage_arrays = jnp.transpose(jnp.stack([global_ids, parent_ids], axis=1), (0, 2, 1))
  else:
    lineage_arrays = None

  initial_weights = jnp.full(
    config.smc_config.population_size,
    1.0 / config.smc_config.population_size,
    dtype=jnp.float32,
  )
  initial_blackjax_states = vmap(
    lambda p: BlackjaxSMCState(
      particles=p,
      weights=initial_weights,  # `initial_weights` is broadcasted
      update_parameters=jnp.zeros(0, dtype=jnp.float32),
    ),
  )(template_population_stacked)

  # --- Refactored: Direct Batched State Construction ---
  # Construct the batched IslandState Pytree directly.
  # Each field will have a leading dimension of size `n_islands`.
  initial_island_states = IslandState(
    key=island_keys,
    blackjax_state=initial_blackjax_states,
    beta=jnp.array(config.island_betas, dtype=jnp.float32),
    logZ_estimate=jnp.zeros(config.n_islands, dtype=jnp.float32),
    mean_fitness=jnp.zeros(config.n_islands, dtype=jnp.float32),
    ess=jnp.zeros(config.n_islands, dtype=jnp.float32),
    step=jnp.zeros(config.n_islands, dtype=jnp.int32),
    lineage=lineage_arrays,
  )

  return PRSMCState(
    prng_key=key,
    current_overall_state=initial_island_states,
    total_swaps_attempted=jnp.array(0, dtype=jnp.int32),
    total_swaps_accepted=jnp.array(0, dtype=jnp.int32),
  )


@partial(jit, static_argnames=("config", "fitness_fn"))
def island_smc_step(
  island_state: IslandState,
  config: ParallelReplicaConfig,
  fitness_fn: StackedFitnessFn,
) -> IslandState:
  """Perform one SMC step (mutation, weighting, UNCONDITIONAL resampling) for a single island.

  CRITICAL: This step now uses the native Blackjax flow: Resample(t-1) -> Update(t) -> Weight(t)

  Args:
    island_state: Current state of the island.
    config: Configuration for the SMC step.
    fitness_fn: Stacked fitness function.

  Returns:
    Updated state of the island.

  """
  key_step, key_next_island = jax.random.split(island_state.key, 2)
  key_smc_update, key_smc_weight, key_smc_resample = jax.random.split(key_step, 3)

  p_step = island_state.step

  @partial(
    jit,
    static_argnames=("p_step", "mutation_rate", "q_states"),
  )
  def mutation_update_fn(
    keys: PRNGKeyArray,
    particles: Array,
    _update_parameters: Array,
    p_step: Int,
    mutation_rate: Float,
    q_states: Int,
  ) -> tuple[Array, None]:
    """Apply mutation to the resampled particles."""
    mutated_particles = jax.lax.cond(
      (p_step > 0),
      lambda x: mutate(
        keys,
        x,
        mutation_rate,
        q_states,
      ).astype(x.dtype),
      lambda x: x,
      particles,
    )

    return mutated_particles, None

  @partial(jit, static_argnames=("fitness_fn_partial", "beta"))
  def weight_fn(
    sequence: Array,
    fitness_fn_partial: Callable,
    beta: Float,
  ) -> Array:
    """Weight function for the SMC step."""
    fitness_values, _ = fitness_fn_partial(sequence)

    return jnp.asarray(
      jnp.where(
        jnp.isneginf(fitness_values),
        -jnp.inf,
        beta * fitness_values,
      ),
    )

  sequence_alphabet_size = 4 if config.sequence_type == "nucleotide" else 20

  mutation_partial = partial(
    mutation_update_fn,
    p_step=p_step,
    mutation_rate=config.mutation_rate,
    sequence_alphabet_size=sequence_alphabet_size,
  )

  fitness_fn_partial = partial(fitness_fn, key_smc_weight, None)
  weight_partial = partial(
    weight_fn,
    fitness_fn_partial=fitness_fn_partial,
    beta=island_state.beta,
  )

  next_blackjax_state, info = blackjax_smc_step(
    rng_key=key_step,
    state=island_state.blackjax_state,
    update_fn=mutation_partial,
    weight_fn=weight_partial,
    resample_fn=partial(resample, config.smc_config),
  )

  normalized_weights = next_blackjax_state.weights
  ess = 1.0 / jnp.sum(normalized_weights**2)

  next_lineage = None
  if config.track_lineage and island_state.lineage is not None:
    previous_lineage = island_state.lineage

    parent_global_ids = previous_lineage[info.ancestors, 0]

    total_pop_size = config.smc_config.population_size * config.n_islands
    new_global_ids = jnp.arange(
      p_step * total_pop_size,
      p_step * total_pop_size + config.smc_config.population_size,
      dtype=jnp.int32,
    )

    next_lineage = jnp.stack([new_global_ids, parent_global_ids], axis=1)

  fitness_values, _ = fitness_fn(
    jnp.asarray(next_blackjax_state.particles),
    key_smc_weight,
    None,
  )

  finite_fitness_mask = jnp.isfinite(fitness_values)
  eps = 1e-9

  if isinstance(normalized_weights, jax.Array) and isinstance(finite_fitness_mask, jax.Array):
    weighted_finite_fitness = jnp.where(
      finite_fitness_mask,
      fitness_values * normalized_weights,
      0.0,
    )
    sum_weights_for_finite = jnp.sum(jnp.where(finite_fitness_mask, normalized_weights, 0.0))
  else:
    weighted_finite_fitness = jnp.array(0.0)
    sum_weights_for_finite = jnp.array(0.0)

  if not isinstance(weighted_finite_fitness, jax.Array):
    msg = f"Expected weighted_finite_fitness to be a jax.Array, got {type(weighted_finite_fitness)}"
    raise TypeError(
      msg,
    )

  mean_fitness_this_step = jax.lax.cond(
    sum_weights_for_finite > eps,
    lambda: jnp.sum(weighted_finite_fitness) / sum_weights_for_finite,
    lambda: jnp.nan,
  )

  if not isinstance(mean_fitness_this_step, jax.Array):
    msg = f"Expected mean_fitness_this_step to be a jax.Array, got {type(mean_fitness_this_step)}"
    raise TypeError(
      msg,
    )

  if not isinstance(fitness_values, jax.Array):
    msg = f"Expected fitness_values to be a jax.Array, got {type(fitness_values)}"
    raise TypeError(msg)

  max_fitness_this_step = jnp.max(jnp.where(jnp.isfinite(fitness_values), fitness_values, -jnp.inf))
  max_fitness_this_step = jnp.where(
    jnp.all(~jnp.isfinite(fitness_values)),
    jnp.nan,
    max_fitness_this_step,
  )

  return IslandState(
    key=key_next_island,
    beta=island_state.beta,
    blackjax_state=next_blackjax_state,
    smc_info=info,
    logZ_estimate=island_state.logZ_estimate + info.log_likelihood_increment,
    mean_fitness=mean_fitness_this_step,
    ess=ess,
    step=island_state.step + 1,
    lineage=next_lineage,
  )


@partial(jit, static_argnames=("config", "fitness_fn"))
def prsmc_step(
  island_states: IslandState,
  config: ParallelReplicaConfig,
  fitness_fn: StackedFitnessFn,
) -> tuple[IslandState, tuple[Float, Float, Float, Float]]:
  """Perform a single step of the PRSMC algorithm across all islands.

  Args:
    island_states: Current state of all islands.
    config: Configuration for the SMC step.
    fitness_fn: Stacked fitness function.

  Returns:
    tuple containing:
      - Updated island states
      - Tuple of metrics (ESS, mean fitness, max fitness, logZ increment)

  """
  updated_islands = vmap(
    lambda state: island_smc_step(state, config, fitness_fn),
    in_axes=0,
    out_axes=0,
  )(island_states)

  ess = updated_islands.ess
  mean_fitness = updated_islands.mean_fitness
  max_fitness = jnp.max(
    jnp.asarray(
      jnp.where(
        jnp.isfinite(updated_islands.mean_fitness),
        updated_islands.mean_fitness,
        -jnp.inf,
      ),
    ),
    axis=0,
  )
  logZ_increment = updated_islands.logZ_estimate - island_states.logZ_estimate  # noqa: N806

  return updated_islands, (ess, mean_fitness, max_fitness, logZ_increment)


@partial(jit, static_argnames=("config", "fitness_fn"))
def migrate(
  island_states: IslandState,
  meta_beta_current_value: Float,
  key_exchange: PRNGKeyArray,
  config: ExchangeConfig,
  fitness_fn: StackedFitnessFn,
) -> tuple[IslandState, Int]:
  """Perform replica exchange attempts. Acceptance is scaled by meta_beta_current_value.

  Args:
    island_states: Current state of the islands.
    meta_beta_current_value: Current value of the meta beta parameter.
    key_exchange: JAX PRNG key for random operations.
    config: Configuration for the exchange process.
    fitness_fn: Stacked fitness function.

  Returns:
    tuple containing updated island states and total number of accepted swaps.

  """
  all_population_stacked = island_states.blackjax_state.particles
  all_betas = island_states.beta
  mean_fitness_per_island = island_states.mean_fitness

  all_lineage_stacked = island_states.lineage

  num_swaps_accepted_total = jnp.array(0, dtype=jnp.int32)

  safe_mean_fitness = jnp.nan_to_num(mean_fitness_per_island, nan=0.0)
  min_fitness_val = jnp.min(safe_mean_fitness)
  shifted_fitness = safe_mean_fitness - min_fitness_val + 1e-9
  sum_shifted_fitness = jnp.sum(shifted_fitness)
  eps = 1e-9
  probs_idx1 = jax.lax.cond(
    (sum_shifted_fitness > eps) & (config.n_islands > 0),
    lambda: shifted_fitness / sum_shifted_fitness,
    lambda: jnp.ones(config.n_islands, dtype=jnp.float32) / jnp.maximum(config.n_islands, 1.0),
  )

  def attempt_exchange(_: int, loop_state: tuple) -> tuple:
    key_attempt, current_population_state, current_lineage_state, current_accepted_swaps = (
      loop_state
    )
    (
      key_select_idx1,
      key_select_idx2,
      key_particle_choice1,
      key_particle_choice2,
      key_acceptance,
      key_next_attempt,
    ) = jax.random.split(key_attempt, 6)
    idx1 = jax.random.choice(key_select_idx1, jnp.arange(config.n_islands), p=probs_idx1)
    offset_for_idx2 = jax.random.randint(
      key_select_idx2,
      shape=(),
      minval=1,
      maxval=jnp.maximum(config.n_islands, 2),
    )
    idx2 = (idx1 + offset_for_idx2) % config.n_islands
    idx2 = jax.lax.cond(config.n_islands <= 1, lambda: idx1, lambda: idx2)

    island_a_population = current_population_state[idx1]
    island_b_population = current_population_state[idx2]

    island_a_lineage = current_lineage_state[idx1]
    island_b_lineage = current_lineage_state[idx2]

    beta_a = all_betas[idx1]
    beta_b = all_betas[idx2]
    sequence_idx_a = jax.random.randint(
      key_particle_choice1,
      shape=(),
      minval=0,
      maxval=config.population_size_per_island,
    )
    sequence_idx_b = jax.random.randint(
      key_particle_choice2,
      shape=(),
      minval=0,
      maxval=config.population_size_per_island,
    )
    config_a = island_a_population[sequence_idx_a]
    config_b = island_b_population[sequence_idx_b]

    fitness_a, _ = fitness_fn(
      jnp.expand_dims(config_a, 0),
      key_acceptance,
      None,
    )
    fitness_b, _ = fitness_fn(
      jnp.expand_dims(config_b, 0),
      key_acceptance,
      None,
    )
    fitness_a = jnp.mean(fitness_a)
    fitness_b = jnp.mean(fitness_b)

    log_acceptance_ratio = meta_beta_current_value * (beta_a - beta_b) * (fitness_b - fitness_a)
    log_acceptance_ratio = jnp.where(
      jnp.isinf(fitness_a) | jnp.isinf(fitness_b) | (idx1 == idx2),
      -jnp.inf,
      log_acceptance_ratio,
    )
    accept = (
      jnp.log(jax.random.uniform(key_acceptance, shape=(), minval=1e-38, maxval=1.0))
      < log_acceptance_ratio
    )

    new_population_state = jax.lax.cond(
      accept,
      lambda parts: parts.at[idx1, sequence_idx_a]
      .set(config_b)
      .at[idx2, sequence_idx_b]
      .set(config_a),
      lambda parts: parts,
      current_population_state,
    )

    new_lineage_state = jax.lax.cond(
      accept & config.track_lineage,
      lambda lineage_parts: lineage_parts.at[idx1, sequence_idx_a]
      .set(island_b_lineage[sequence_idx_b])
      .at[idx2, sequence_idx_b]
      .set(island_a_lineage[sequence_idx_a]),
      lambda lineage_parts: lineage_parts,
      current_lineage_state,
    )

    updated_accepted_swaps = current_accepted_swaps + jax.lax.select(accept, 1, 0)

    return key_next_attempt, new_population_state, new_lineage_state, updated_accepted_swaps

  initial_loop_state = (
    key_exchange,
    all_population_stacked,
    all_lineage_stacked,
    num_swaps_accepted_total,
  )

  final_population_state, final_lineage_state, total_accepted_swaps = jax.lax.cond(
    (config.n_exchange_attempts > 0) & (config.n_islands > 1),
    lambda: lax.fori_loop(0, config.n_exchange_attempts, attempt_exchange, initial_loop_state)[1:],
    lambda: (all_population_stacked, all_lineage_stacked, num_swaps_accepted_total),
  )

  updated_blackjax_states = island_states.blackjax_state._replace(
    particles=final_population_state,
  )

  updated_island_states = IslandState(
    key=island_states.key,
    beta=island_states.beta,
    logZ_estimate=island_states.logZ_estimate,
    mean_fitness=island_states.mean_fitness,
    ess=island_states.ess,
    step=island_states.step,
    blackjax_state=updated_blackjax_states,
    lineage=final_lineage_state,
  )

  return updated_island_states, total_accepted_swaps


def initialize_prsmc_state(config: ParallelReplicaConfig, key: PRNGKeyArray) -> PRSMCState:
  """Initialize the state for the Parallel Replica SMC sampler in a JAX-idiomatic way."""
  key_init_islands, key_smc_loop = jax.random.split(key)

  from proteinsmc.utils.constants import AA_CHAR_TO_INT_MAP, NUCLEOTIDES_INT_MAP

  if config.sequence_type == "protein":
    try:
      initial_seq = jnp.array(
        [AA_CHAR_TO_INT_MAP[res] for res in config.seed_sequence],
        dtype=jnp.int8,
      )
    except KeyError as e:
      msg = f"Invalid amino acid: {e.args[0]}"
      raise ValueError(msg) from e
  else:  # nucleotide
    try:
      initial_seq = jnp.array(
        [NUCLEOTIDES_INT_MAP[nuc] for nuc in config.seed_sequence],
        dtype=jnp.int8,
      )
    except KeyError as e:
      msg = f"Invalid nucleotide: {e.args[0]}"
      raise ValueError(msg) from e

  population_size = config.smc_config.population_size * config.n_islands
  initial_population = jnp.tile(initial_seq, (population_size, 1))

  initial_population = diversify_initial_sequences(
    key=key_init_islands,
    seed_sequences=initial_population,
    mutation_rate=config.diversification_ratio,
    sequence_type=config.sequence_type,  # type: ignore[arg-type]
  )
  initial_populations_stacked = initial_population.reshape(
    config.n_islands,
    config.smc_config.population_size,
    -1,
  )

  island_keys = jax.random.split(key_init_islands, config.n_islands)
  island_betas = jnp.array(config.island_betas, dtype=jnp.float32)

  initial_weights = jnp.full(
    config.smc_config.population_size,
    1.0 / config.smc_config.population_size,
    dtype=jnp.float32,
  )
  initial_blackjax_states = vmap(
    lambda p: BlackjaxSMCState(
      particles=p,
      weights=initial_weights,
      update_parameters=jnp.array(0.0, dtype=jnp.float32),
    ),
  )(initial_populations_stacked)

  if config.track_lineage:
    island_pop_size = config.smc_config.population_size
    island_indices = jnp.arange(config.n_islands)[:, None]
    particle_indices = jnp.arange(island_pop_size)[None, :]
    global_ids = island_indices * island_pop_size + particle_indices
    parent_ids = jnp.full_like(global_ids, -1)
    initial_lineage_arrays = jnp.transpose(jnp.stack([global_ids, parent_ids], axis=1), (0, 2, 1))
  else:
    initial_lineage_arrays = None

  initial_island_states = IslandState(
    key=island_keys,
    beta=island_betas,
    logZ_estimate=jnp.zeros(config.n_islands, dtype=jnp.float32),
    mean_fitness=jnp.zeros(config.n_islands, dtype=jnp.float32),
    ess=jnp.zeros(config.n_islands, dtype=jnp.float32),
    blackjax_state=initial_blackjax_states,
    lineage=initial_lineage_arrays,
    step=jnp.zeros(config.n_islands, dtype=jnp.int32),
  )

  return PRSMCState(
    current_overall_state=initial_island_states,
    prng_key=key_smc_loop,
    total_swaps_attempted=jnp.array(0, dtype=jnp.int32),
    total_swaps_accepted=jnp.array(0, dtype=jnp.int32),
  )


def run_prsmc_loop(
  config: ParallelReplicaConfig,
  initial_state: PRSMCState,
  fitness_fn: StackedFitnessFn,
  annealing_fn: AnnealingFn,
) -> tuple[PRSMCState, dict]:
  """JIT-compiled Parallel Replica SMC loop."""

  @partial(jit, static_argnames=("step_config", "fitness_fn", "annealing_fn"))
  def _parallel_replica_scan_step(
    carry_state: PRSMCState,
    step_idx: Int,
    config: ParallelReplicaConfig,
    fitness_fn: StackedFitnessFn,
    annealing_fn: AnnealingFn,
  ) -> tuple[PRSMCState, dict]:
    key_step, next_smc_loop_key = jax.random.split(carry_state.prng_key)

    current_overall_state, island_metrics = prsmc_step(
      carry_state.current_overall_state,
      config,
      fitness_fn,
    )
    ess_p, mean_fit_p, max_fit_p, logZ_inc_p = island_metrics  # noqa: N806
    current_meta_beta = annealing_fn(
      current_step=step_idx,  # type: ignore[call-arg]
      _context=None,  # No context needed for this annealing function
    )

    exchange_config = config.exchange_config
    do_exchange = (step_idx + 1) % exchange_config.exchange_frequency == 0

    current_overall_state, num_accepted_this_cycle = lax.cond(
      do_exchange & (exchange_config.n_islands > 1),
      lambda: migrate(
        island_states=current_overall_state,
        meta_beta_current_value=current_meta_beta,
        key_exchange=key_step,
        config=exchange_config,
        fitness_fn=fitness_fn,
      ),
      lambda: (current_overall_state, jnp.array(0, dtype=jnp.int32)),
    )

    total_swaps_attempted_this_cycle = lax.cond(
      do_exchange & (exchange_config.n_islands > 1),
      lambda: jnp.array(exchange_config.n_exchange_attempts, dtype=jnp.int32),
      lambda: jnp.array(0, dtype=jnp.int32),
    )

    next_carry_state = PRSMCState(
      current_overall_state=current_overall_state,
      prng_key=next_smc_loop_key,
      total_swaps_attempted=carry_state.total_swaps_attempted + total_swaps_attempted_this_cycle,
      total_swaps_accepted=carry_state.total_swaps_accepted + num_accepted_this_cycle,
    )

    collected_metrics = {
      "ess_per_island": ess_p,
      "mean_fitness_per_island": mean_fit_p,
      "max_fitness_per_island": max_fit_p,
      "logZ_increment_per_island": logZ_inc_p,
      "meta_beta": current_meta_beta,
      "num_accepted_swaps": num_accepted_this_cycle,
      "num_attempted_swaps": total_swaps_attempted_this_cycle,
      "population": current_overall_state.population,  # Sequences for this step
      "lineage": current_overall_state.lineage,  # Lineage metadata for this step
    }
    return next_carry_state, collected_metrics

  final_carry_state, collected_outputs_scan = lax.scan(
    lambda carry, scan_input: _parallel_replica_scan_step(
      carry,
      scan_input,
      config,
      fitness_fn,
      annealing_fn,
    ),
    initial_state,
    jnp.arange(config.num_samples, dtype=jnp.int32),
    length=config.num_samples,
  )

  return final_carry_state, collected_outputs_scan
