from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import jit, lax, vmap
from jax import random as jax_random

from ..utils.annealing_schedules import (
  cosine_schedule_py,
  exponential_schedule_py,
  linear_schedule_py,
  static_schedule_py,
)
from ..utils.helper_functions import mutation_kernel_jax, resampling_kernel_jax


# --- Parallel Replica SMC JAX ---
class IslandState(NamedTuple):
  key: jax.random.PRNGKey
  particles: jnp.ndarray
  beta: jnp.float32
  logZ_estimate: jnp.float32


@partial(
  jit,
  static_argnames=(
    "protein_length",
    "n_nuc_alphabet_size",
    "n_particles_per_island",
    "do_initial_mutation",
    "ess_threshold_frac",
  ),
)
def smc_step_for_island_jax(
  island_state,
  p_step,
  fitness_calculation_population_fn,
  mu_nuc,
  protein_length,
  n_nuc_alphabet_size,
  n_particles_per_island,
  ess_threshold_frac,
  do_initial_mutation=False,
):
  """
  Performs one SMC step (mutation, weighting, UNCONDITIONAL resampling)     for a single island.
  ESS is calculated for metrics only. ess_threshold_frac is not used for     resampling decision.
  """
  key_mutate, key_resample, key_next_island = jax_random.split(island_state.key, 3)

  current_particles = island_state.particles

  # 1. Mutation (M_kernel)
  mutated_particles = jax.lax.cond(
    (p_step > 0) | do_initial_mutation,
    lambda x: mutation_kernel_jax(key_mutate, x, mu_nuc, protein_length, n_nuc_alphabet_size),
    lambda x: x,  # No mutation
    current_particles,
  )

  # 2. Fitness Calculation & Weighting (lG_kernel)
  # Use the passed-in fitness_calculation_population_fn
  fitness_values = fitness_calculation_population_fn(mutated_particles)

  log_potential_values = jnp.where(
    jnp.isneginf(fitness_values), -jnp.inf, island_state.beta * fitness_values
  )
  log_weights = log_potential_values

  # 3. UNCONDITIONAL Resampling
  final_particles_for_step, ess, normalized_weights = resampling_kernel_jax(
    key_resample, mutated_particles, log_weights, n_particles_per_island
  )

  # Metrics for this step for this island
  finite_fitness_mask = jnp.isfinite(fitness_values)
  weighted_finite_fitness = jnp.where(finite_fitness_mask, fitness_values * normalized_weights, 0.0)
  sum_weights_for_finite = jnp.sum(jnp.where(finite_fitness_mask, normalized_weights, 0.0))

  mean_fitness_this_step = jax.lax.cond(
    sum_weights_for_finite > 1e-9,
    lambda: jnp.sum(weighted_finite_fitness) / sum_weights_for_finite,
    lambda: jnp.nan,
  )

  max_fitness_this_step = jnp.max(jnp.where(jnp.isfinite(fitness_values), fitness_values, -jnp.inf))
  max_fitness_this_step = jnp.where(
    jnp.all(~jnp.isfinite(fitness_values)), jnp.nan, max_fitness_this_step
  )

  # Update LogZ estimate
  valid_log_weights = jnp.where(jnp.isneginf(log_weights), -jnp.inf, log_weights)
  max_l_w = jnp.max(valid_log_weights)
  safe_max_l_w = jnp.where(jnp.isneginf(max_l_w), 0.0, max_l_w)
  log_sum_exp_weights = safe_max_l_w + jnp.log(jnp.sum(jnp.exp(valid_log_weights - safe_max_l_w)))

  current_logZ_increment = log_sum_exp_weights - jnp.log(jnp.maximum(n_particles_per_island, 1.0))
  current_logZ_increment = jnp.where(
    jnp.isneginf(log_sum_exp_weights) | (n_particles_per_island == 0),
    -jnp.inf,
    current_logZ_increment,
  )

  updated_island_state = IslandState(
    key=key_next_island,
    particles=final_particles_for_step,
    beta=island_state.beta,
    logZ_estimate=island_state.logZ_estimate + current_logZ_increment,
  )
  return updated_island_state, (
    ess,
    mean_fitness_this_step,
    max_fitness_this_step,
    current_logZ_increment,
  )


vmapped_smc_step = jit(
  vmap(
    smc_step_for_island_jax,
    in_axes=(IslandState(0, 0, 0, 0), None, None, None, None, None, None, None, None),
    out_axes=(IslandState(0, 0, 0, 0), (0, 0, 0, 0)),
  ),
  static_argnames=(
    "protein_length",
    "n_nuc_alphabet_size",
    "n_particles_per_island",
    "ess_threshold_frac",
    "do_initial_mutation",
  ),
)


@partial(
  jit,
  static_argnames=(
    "protein_length",
    "n_nuc_alphabet_size",
    "n_particles_per_island",
    "n_islands",
    "n_exchange_attempts",
  ),
)
def replica_exchange_step_jax(
  current_island_states_tuple: IslandState,
  mean_fitness_per_island: jnp.ndarray,
  meta_beta_current_value: jnp.float32,
  key_exchange: jax.random.PRNGKey,
  fitness_calculation_single_fn: Callable,
  protein_length: int,
  n_nuc_alphabet_size: int,
  n_particles_per_island: int,
  n_islands: int,
  n_exchange_attempts: int,
):
  """
  Performs replica exchange attempts. Acceptance is scaled by meta_beta_current_value.
  """
  all_particles_stacked = current_island_states_tuple.particles
  all_betas = current_island_states_tuple.beta
  num_swaps_accepted_total = jnp.array(0, dtype=jnp.int32)

  safe_mean_fitness = jnp.nan_to_num(mean_fitness_per_island, nan=0.0)
  min_fitness_val = jnp.min(safe_mean_fitness)
  shifted_fitness = safe_mean_fitness - min_fitness_val + 1e-9
  sum_shifted_fitness = jnp.sum(shifted_fitness)
  probs_idx1 = jax.lax.cond(
    (sum_shifted_fitness > 1e-9) & (n_islands > 0),
    lambda: shifted_fitness / sum_shifted_fitness,
    lambda: jnp.ones(n_islands, dtype=jnp.float32) / jnp.maximum(n_islands, 1.0),
  )

  def exchange_attempt_loop_body(attempt_idx, loop_state):
    key_attempt, current_particles_state, current_accepted_swaps = loop_state
    (
      key_select_idx1,
      key_select_idx2,
      key_particle_choice1,
      key_particle_choice2,
      key_acceptance,
      key_next_attempt,
    ) = jax_random.split(key_attempt, 6)
    idx1 = jax_random.choice(key_select_idx1, jnp.arange(n_islands), p=probs_idx1)
    offset_for_idx2 = jax_random.randint(
      key_select_idx2, shape=(), minval=1, maxval=jnp.maximum(n_islands, 2)
    )
    idx2 = (idx1 + offset_for_idx2) % n_islands
    idx2 = jax.lax.cond(n_islands <= 1, lambda: idx1, lambda: idx2)
    island_a_particles = current_particles_state[idx1]
    island_b_particles = current_particles_state[idx2]
    beta_a = all_betas[idx1]
    beta_b = all_betas[idx2]
    particle_idx_a = jax_random.randint(
      key_particle_choice1, shape=(), minval=0, maxval=n_particles_per_island
    )
    particle_idx_b = jax_random.randint(
      key_particle_choice2, shape=(), minval=0, maxval=n_particles_per_island
    )
    config_a = island_a_particles[particle_idx_a]
    config_b = island_b_particles[particle_idx_b]

    # Use the passed-in single fitness calculation function
    fitness_a = fitness_calculation_single_fn(config_a)
    fitness_b = fitness_calculation_single_fn(config_b)

    # *** MODIFIED acceptance probability with meta_beta_current_value ***
    log_acceptance_ratio = meta_beta_current_value * (beta_a - beta_b) * (fitness_b - fitness_a)
    log_acceptance_ratio = jnp.where(
      jnp.isinf(fitness_a) | jnp.isinf(fitness_b) | (idx1 == idx2), -jnp.inf, log_acceptance_ratio
    )
    accept = (
      jnp.log(jax_random.uniform(key_acceptance, shape=(), minval=1e-38, maxval=1.0))
      < log_acceptance_ratio
    )
    new_particles_state = jax.lax.cond(
      accept,
      lambda parts: parts.at[idx1, particle_idx_a]
      .set(config_b)
      .at[idx2, particle_idx_b]
      .set(config_a),
      lambda parts: parts,
      current_particles_state,
    )
    updated_accepted_swaps = current_accepted_swaps + jax.lax.select(accept, 1, 0)
    return key_next_attempt, new_particles_state, updated_accepted_swaps

  initial_loop_state = (key_exchange, all_particles_stacked, num_swaps_accepted_total)
  final_particles_state, total_accepted_swaps = jax.lax.cond(
    (n_exchange_attempts > 0) & (n_islands > 1),
    lambda: lax.fori_loop(0, n_exchange_attempts, exchange_attempt_loop_body, initial_loop_state)[
      1:
    ],
    lambda: (all_particles_stacked, num_swaps_accepted_total),
  )
  updated_island_states = current_island_states_tuple._replace(particles=final_particles_state)
  return updated_island_states, total_accepted_swaps


def run_parallel_replica_smc_jax(
  master_sim_key,
  protein_length: int,
  n_nuc_alphabet_size: int,
  n_islands: int,
  island_betas: list[float],
  n_particles_per_island: int,
  n_smc_steps: int,
  exchange_frequency: int,
  n_exchange_attempts_per_cycle: int,
  fitness_calculation_population_fn: Callable,  # Callable for population fitness calculation
  fitness_calculation_single_fn: Callable,  # Callable for single fitness calculation
  initial_particle_generation_fn: Callable,  # Callable for initial particle generation
  mu_nuc: float,  # Nucleotide mutation rate
  ess_threshold_fraction: float = 0.5,
  meta_beta_schedule_type: str = "static",
  meta_beta_max_val: float = 1.0,
  meta_beta_schedule_rate: float = 5.0,
):
  print(
    f"Running Parallel Replica SMC (JAX) - Meta-Annealing: "
    f"{meta_beta_schedule_type.upper()}, MaxMetaBeta: "
    f"{meta_beta_max_val:.2f} - UNCONDITIONAL Resampling."
  )
  print(
    f"L={protein_length}, Q={n_nuc_alphabet_size}, Islands={n_islands}, "
    f"Particles/Island={n_particles_per_island}, Steps={n_smc_steps}"
  )

  key_init_islands, key_smc_loop, key_next_master = jax_random.split(master_sim_key, 3)

  # Generate initial particles using the provided callable
  initial_particles_array = initial_particle_generation_fn(
    key_init_islands, n_islands, n_particles_per_island, protein_length, n_nuc_alphabet_size
  )

  island_keys = jax_random.split(key_init_islands, n_islands)
  initial_island_states = IslandState(
    key=island_keys,
    particles=initial_particles_array,
    beta=jnp.array(island_betas, dtype=jnp.float32),
    logZ_estimate=jnp.zeros(n_islands, dtype=jnp.float32),
  )

  # Define the JAX-jitted SMC step function for jax.lax.scan
  @partial(
    jit,
    static_argnames=(
      "protein_length_static",
      "n_nuc_alphabet_size_static",
      "n_particles_per_island_static",
      "n_islands_static",
      "exchange_frequency_static",
      "n_exchange_attempts_per_cycle_static",
      "ess_threshold_fraction_static",
      "meta_beta_schedule_type_static",
      "meta_beta_max_val_static",
      "meta_beta_schedule_rate_static",
    ),
  )
  def _parallel_replica_scan_step(
    carry_state,
    scan_inputs_current_step,
    protein_length_static,
    n_nuc_alphabet_size_static,
    n_particles_per_island_static,
    n_islands_static,
    exchange_frequency_static,
    n_exchange_attempts_per_cycle_static,
    ess_threshold_fraction_static,
    meta_beta_schedule_type_static,
    meta_beta_max_val_static,
    meta_beta_schedule_rate_static,
  ):
    current_overall_state, current_smc_loop_key, total_swaps_accepted, total_swaps_attempted = (
      carry_state
    )
    p_step = scan_inputs_current_step

    key_step_smc, key_step_exchange, next_smc_loop_key = jax_random.split(current_smc_loop_key, 3)

    current_overall_state, island_metrics_this_step = vmapped_smc_step(
      current_overall_state,
      p_step,
      fitness_calculation_population_fn,
      mu_nuc,
      protein_length_static,
      n_nuc_alphabet_size_static,
      n_particles_per_island_static,
      ess_threshold_fraction_static,
      False,
    )
    ess_p, mean_fit_p, max_fit_p, logZ_inc_p = island_metrics_this_step

    # Calculate current meta_beta based on schedule
    current_step_1_indexed = p_step + 1
    current_meta_beta_py_scalar = jax.lax.switch(
      jnp.array(0, dtype=jnp.int32),  # Default to 0 for linear
      [
        lambda: linear_schedule_py(current_step_1_indexed, n_smc_steps, meta_beta_max_val_static),
        lambda: exponential_schedule_py(
          current_step_1_indexed,
          n_smc_steps,
          meta_beta_max_val_static,
          rate=meta_beta_schedule_rate_static,
        ),
        lambda: cosine_schedule_py(current_step_1_indexed, n_smc_steps, meta_beta_max_val_static),
        lambda: static_schedule_py(current_step_1_indexed, n_smc_steps, meta_beta_max_val_static),
      ],
      jnp.where(
        meta_beta_schedule_type_static == "linear",
        0,
        jnp.where(
          meta_beta_schedule_type_static == "exponential",
          1,
          jnp.where(
            meta_beta_schedule_type_static == "cosine",
            2,
            jnp.where(meta_beta_schedule_type_static == "static", 3, 0),
          ),
        ),
      ),  # Default to linear if unknown
    )
    current_meta_beta_jax = jnp.float32(current_meta_beta_py_scalar)

    num_accepted_this_cycle = jnp.array(0, dtype=jnp.int32)
    total_swaps_attempted_this_cycle = jnp.array(0, dtype=jnp.int32)

    do_exchange = (p_step + 1) % exchange_frequency_static == 0

    current_overall_state, num_accepted_this_cycle = jax.lax.cond(
      do_exchange & (n_islands_static > 1),
      lambda: replica_exchange_step_jax(
        current_island_states_tuple=current_overall_state,
        mean_fitness_per_island=mean_fit_p,
        meta_beta_current_value=current_meta_beta_jax,
        key_exchange=key_step_exchange,
        fitness_calculation_single_fn=fitness_calculation_single_fn,
        protein_length=protein_length_static,
        n_nuc_alphabet_size=n_nuc_alphabet_size_static,
        n_particles_per_island=n_particles_per_island_static,
        n_islands=n_islands_static,
        n_exchange_attempts=n_exchange_attempts_per_cycle_static,
      ),
      lambda: (current_overall_state, jnp.array(0, dtype=jnp.int32)),
    )
    total_swaps_attempted_this_cycle = jax.lax.cond(
      do_exchange & (n_islands_static > 1),
      lambda: jnp.array(n_exchange_attempts_per_cycle_static, dtype=jnp.int32),
      lambda: jnp.array(0, dtype=jnp.int32),
    )

    total_swaps_accepted += num_accepted_this_cycle
    total_swaps_attempted += total_swaps_attempted_this_cycle

    next_carry_state = (
      current_overall_state,
      next_smc_loop_key,
      total_swaps_accepted,
      total_swaps_attempted,
    )

    collected_metrics = {
      "ess_per_island": ess_p,
      "mean_fitness_per_island": mean_fit_p,
      "max_fitness_per_island": max_fit_p,
      "logZ_increment_per_island": logZ_inc_p,
      "meta_beta": current_meta_beta_py_scalar,
      "num_accepted_swaps": num_accepted_this_cycle,
      "num_attempted_swaps": total_swaps_attempted_this_cycle,
    }
    return next_carry_state, collected_metrics

  initial_carry = (
    initial_island_states,
    key_smc_loop,
    jnp.array(0, dtype=jnp.int32),
    jnp.array(0, dtype=jnp.int32),
  )

  scan_over_ijnputs = jnp.arange(n_smc_steps)

  final_carry_state, collected_outputs_scan = lax.scan(
    lambda carry, scan_in: _parallel_replica_scan_step(
      carry,
      scan_in,
      protein_length,
      n_nuc_alphabet_size,
      n_particles_per_island,
      n_islands,
      exchange_frequency,
      n_exchange_attempts_per_cycle,
      ess_threshold_fraction,
      meta_beta_schedule_type,
      meta_beta_max_val,
      meta_beta_schedule_rate,
    ),
    initial_carry,
    scan_over_ijnputs,
    length=n_smc_steps,
  )

  final_island_states, _, final_total_swaps_accepted, final_total_swaps_attempted = (
    final_carry_state
  )

  # Unpack collected outputs
  history_ess_per_island = collected_outputs_scan["ess_per_island"]
  history_mean_fitness_per_island = collected_outputs_scan["mean_fitness_per_island"]
  history_max_fitness_per_island = collected_outputs_scan["max_fitness_per_island"]
  history_logZ_increment_per_island = collected_outputs_scan["logZ_increment_per_island"]
  history_meta_beta = collected_outputs_scan["meta_beta"]
  history_num_accepted_swaps = collected_outputs_scan["num_accepted_swaps"]
  history_num_attempted_swaps = collected_outputs_scan["num_attempted_swaps"]

  final_logZ_estimates_per_island = final_island_states.logZ_estimate
  swap_acceptance_rate = jnp.where(
    final_total_swaps_attempted > 0,
    final_total_swaps_accepted / final_total_swaps_attempted,
    jnp.array(0.0, dtype=jnp.float32),
  )

  print(
    f"Finished Parallel Replica SMC (JAX) - Meta-Annealing: {meta_beta_schedule_type.upper()} - UNCONDITIONAL Resampling."
  )
  results = {
    "protein_length": protein_length,
    "n_nuc_alphabet_size": n_nuc_alphabet_size,
    "mu_nuc": mu_nuc,
    "n_islands": n_islands,
    "island_betas": jnp.array(island_betas),
    "n_particles_per_island": n_particles_per_island,
    "n_smc_steps": n_smc_steps,
    "exchange_frequency": exchange_frequency,
    "n_exchange_attempts_per_cycle": n_exchange_attempts_per_cycle,
    "ess_threshold_fraction": ess_threshold_fraction,
    "meta_beta_schedule_type": meta_beta_schedule_type,
    "meta_beta_max_val": meta_beta_max_val,
    "meta_beta_schedule_rate": meta_beta_schedule_rate,
    "final_logZ_estimates_per_island": final_logZ_estimates_per_island,
    "swap_acceptance_rate": swap_acceptance_rate,
    "history_mean_fitness_per_island": history_mean_fitness_per_island,
    "history_max_fitness_per_island": history_max_fitness_per_island,
    "history_ess_per_island": history_ess_per_island,
    "history_logZ_increment_per_island": history_logZ_increment_per_island,
    "history_meta_beta": history_meta_beta,
    "history_num_accepted_swaps": history_num_accepted_swaps,
    "history_num_attempted_swaps": history_num_attempted_swaps,
  }
  return results
  return results
