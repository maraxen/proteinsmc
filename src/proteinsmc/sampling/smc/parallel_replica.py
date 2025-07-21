"""Implementation of Parallel Replica inspired Sequential Monte Carlo (PRSMC) sampling."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import jit, lax, vmap

if TYPE_CHECKING:
  from jaxtyping import Float, Int, PRNGKeyArray

  from proteinsmc.models.fitness import StackedFitnessFn
  from proteinsmc.utils.annealing import AnnealingFuncSignature
from proteinsmc.models.parallel_replica import (
  ExchangeConfig,
  IslandState,
  ParallelReplicaConfig,
  PRSMCState,
  PRSMCStepConfig,
)
from proteinsmc.sampling.smc.resampling import resample
from proteinsmc.utils import (
  calculate_logZ_increment,
  diversify_initial_sequences,
)
from proteinsmc.utils.mutation import mutate


@partial(jit, static_argnames=("config", "fitness_fn"))
def island_smc_step(
  island_state: IslandState,
  config: PRSMCStepConfig,
  fitness_fn: StackedFitnessFn,
) -> IslandState:
  """Perform one SMC step (mutation, weighting, UNCONDITIONAL resampling) for a single island.

  ESS is calculated for metrics only currently.
  ess_threshold_frac is not used for resampling decision.

  Args:
    island_state: Current state of the island.
    config: Configuration for the SMC step.
    fitness_fn: Stacked fitness function.

  Returns:
    Updated state of the island.

  """
  key_mutate, key_fitness, key_resample, key_next_island = jax.random.split(island_state.key, 4)
  current_population = island_state.population
  p_step = island_state.step
  mutated_population = jax.lax.cond(
    (p_step > 0),
    lambda x: mutate(
      key_mutate,
      x,
      config.mutation_rate,
      4 if config.sequence_type == "nucleotide" else 20,
    ).astype(x.dtype),
    lambda x: x,
    current_population,
  )

  fitness_values, fitness_components = fitness_fn(
    mutated_population,
    key_fitness,  # type: ignore[arg-type]
    _context=None,
  )

  log_potential_values = jnp.where(
    jnp.isneginf(fitness_values),
    -jnp.inf,
    island_state.beta * fitness_values,
  )
  log_weights = log_potential_values

  # Unconditional resampling for now - could be made adaptive in future
  final_population_for_step, ess, normalized_weights = resample(
    key_resample,
    mutated_population,
    log_weights,
  )

  finite_fitness_mask = jnp.isfinite(fitness_values)
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

  eps = 1e-9
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

  current_logZ_increment = calculate_logZ_increment(  # noqa: N806
    log_weights,
    config.population_size_per_island,
  )

  return IslandState(
    key=key_next_island,
    population=final_population_for_step,
    beta=island_state.beta,
    logZ_estimate=island_state.logZ_estimate + current_logZ_increment,
    mean_fitness=mean_fitness_this_step,
    ess=ess,
    step=island_state.step + 1,
  )


@partial(jit, static_argnames=("config", "fitness_fn"))
def prsmc_step(
  island_states: IslandState,
  config: PRSMCStepConfig,
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
  # Apply SMC step to each island
  updated_islands = vmap(
    lambda state: island_smc_step(state, config, fitness_fn),
    in_axes=0,
    out_axes=0,
  )(island_states)

  # Extract metrics
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
  all_population_stacked = island_states.population
  all_betas = island_states.beta
  mean_fitness_per_island = island_states.mean_fitness

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
    key_attempt, current_population_state, current_accepted_swaps = loop_state
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
      key_acceptance,  # type: ignore[arg-type]
      _context=None,
    )
    fitness_b, _ = fitness_fn(
      jnp.expand_dims(config_b, 0),
      key_acceptance,  # type: ignore[arg-type]
      _context=None,
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
    updated_accepted_swaps = current_accepted_swaps + jax.lax.select(accept, 1, 0)
    return key_next_attempt, new_population_state, updated_accepted_swaps

  initial_loop_state = (key_exchange, all_population_stacked, num_swaps_accepted_total)
  final_population_state, total_accepted_swaps = jax.lax.cond(
    (config.n_exchange_attempts > 0) & (config.n_islands > 1),
    lambda: lax.fori_loop(0, config.n_exchange_attempts, attempt_exchange, initial_loop_state)[1:],
    lambda: (all_population_stacked, num_swaps_accepted_total),
  )

  updated_island_states = IslandState(
    key=island_states.key,
    population=final_population_state,
    beta=island_states.beta,
    logZ_estimate=island_states.logZ_estimate,
    mean_fitness=island_states.mean_fitness,
    ess=island_states.ess,
    step=island_states.step,
  )

  return updated_island_states, total_accepted_swaps


def initialize_prsmc_state(config: ParallelReplicaConfig, key: PRNGKeyArray) -> PRSMCState:
  """Initialize the state for the Parallel Replica SMC sampler."""
  key_init_islands, key_smc_loop = jax.random.split(key)

  # Create simple template population based on n_states
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

  population_size = config.step_config.population_size_per_island * config.n_islands
  initial_population = jnp.tile(initial_seq, (population_size, 1))

  initial_population = diversify_initial_sequences(
    key=key_init_islands,
    seed_sequences=initial_population,
    mutation_rate=config.diversification_ratio,
    sequence_type=config.sequence_type,  # type: ignore[arg-type]
  )
  initial_populations = jnp.split(
    initial_population,
    config.n_islands,
    axis=0,
  )

  island_keys = jax.random.split(key_init_islands, config.n_islands)
  island_betas = jnp.array(config.island_betas, dtype=jnp.float32)
  initial_island_states_list = [
    IslandState(
      key=island_keys[i],
      population=initial_populations[i],
      beta=island_betas[i],
      logZ_estimate=jnp.array(0.0, dtype=jnp.float32),
      mean_fitness=jnp.array(0.0, dtype=jnp.float32),
      ess=jnp.array(0.0, dtype=jnp.float32),
    )
    for i in range(config.n_islands)
  ]

  initial_island_states = jax.tree_util.tree_map(
    lambda *xs: jnp.stack(xs, axis=0),
    *initial_island_states_list,
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
  annealing_fn: AnnealingFuncSignature,
) -> tuple[PRSMCState, dict]:
  """JIT-compiled Parallel Replica SMC loop."""

  @partial(jit, static_argnames=("step_config", "fitness_fn", "annealing_fn"))
  def _parallel_replica_scan_step(
    carry_state: PRSMCState,
    step_idx: Int,
    step_config: PRSMCStepConfig,
    fitness_fn: StackedFitnessFn,
    annealing_fn: AnnealingFuncSignature,
  ) -> tuple[PRSMCState, dict]:
    key_step, next_smc_loop_key = jax.random.split(carry_state.prng_key)

    current_overall_state, island_metrics = prsmc_step(
      carry_state.current_overall_state,
      step_config,
      fitness_fn,
    )
    ess_p, mean_fit_p, max_fit_p, logZ_inc_p = island_metrics  # noqa: N806
    current_meta_beta = annealing_fn(
      current_step=step_idx,  # type: ignore[call-arg]
      _context=None,  # No context needed for this annealing function
    )

    exchange_config = step_config.exchange_config
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
    }
    return next_carry_state, collected_metrics

  step_config = config.step_config

  final_carry_state, collected_outputs_scan = lax.scan(
    lambda carry, scan_input: _parallel_replica_scan_step(
      carry,
      scan_input,
      step_config,
      fitness_fn,
      annealing_fn,
    ),
    initial_state,
    jnp.arange(config.num_samples, dtype=jnp.int32),
    length=config.num_samples,
  )

  return final_carry_state, collected_outputs_scan
