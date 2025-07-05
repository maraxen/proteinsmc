from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax import jit, lax, vmap
from jax import random as jax_random
from jaxtyping import PRNGKeyArray

from ..utils import (
  FitnessEvaluator,
  calculate_logZ_increment,
  calculate_population_fitness,
  resample,
)
from ..utils.annealing_schedules import (
  cosine_schedule,
  exponential_schedule,
  linear_schedule,
)
from ..utils.mutate import dispatch_mutation


# --- Output dataclass for results ---
@dataclass
class ParallelReplicaSMCOutput:
  sequence_length: int
  mutation_rate: float
  n_islands: int
  island_betas: jnp.ndarray
  population_size_per_island: int
  n_smc_steps: int
  exchange_frequency: int
  n_exchange_attempts_per_cycle: int
  ess_threshold_fraction: float
  meta_beta_schedule_type: str
  meta_beta_max_val: float
  meta_beta_schedule_rate: float
  final_logZ_estimates_per_island: jnp.ndarray
  swap_acceptance_rate: jnp.ndarray
  history_mean_fitness_per_island: jnp.ndarray
  history_max_fitness_per_island: jnp.ndarray
  history_ess_per_island: jnp.ndarray
  history_logZ_increment_per_island: jnp.ndarray
  history_meta_beta: jnp.ndarray
  history_num_accepted_swaps: jnp.ndarray
  history_num_attempted_swaps: jnp.ndarray

  def tree_flatten(self):
    children = (
      self.island_betas,
      self.final_logZ_estimates_per_island,
      self.swap_acceptance_rate,
      self.history_mean_fitness_per_island,
      self.history_max_fitness_per_island,
      self.history_ess_per_island,
      self.history_logZ_increment_per_island,
      self.history_meta_beta,
      self.history_num_accepted_swaps,
      self.history_num_attempted_swaps,
    )
    aux_data = {
      "sequence_length": self.sequence_length,
      "mutation_rate": self.mutation_rate,
      "n_islands": self.n_islands,
      "population_size_per_island": self.population_size_per_island,
      "n_smc_steps": self.n_smc_steps,
      "exchange_frequency": self.exchange_frequency,
      "n_exchange_attempts_per_cycle": self.n_exchange_attempts_per_cycle,
      "ess_threshold_fraction": self.ess_threshold_fraction,
      "meta_beta_schedule_type": self.meta_beta_schedule_type,
      "meta_beta_max_val": self.meta_beta_max_val,
      "meta_beta_schedule_rate": self.meta_beta_schedule_rate,
    }
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(
      sequence_length=aux_data["sequence_length"],
      mutation_rate=aux_data["mutation_rate"],
      n_islands=aux_data["n_islands"],
      island_betas=children[0],
      population_size_per_island=aux_data["population_size_per_island"],
      n_smc_steps=aux_data["n_smc_steps"],
      exchange_frequency=aux_data["exchange_frequency"],
      n_exchange_attempts_per_cycle=aux_data["n_exchange_attempts_per_cycle"],
      ess_threshold_fraction=aux_data["ess_threshold_fraction"],
      meta_beta_schedule_type=aux_data["meta_beta_schedule_type"],
      meta_beta_max_val=aux_data["meta_beta_max_val"],
      meta_beta_schedule_rate=aux_data["meta_beta_schedule_rate"],
      final_logZ_estimates_per_island=children[1],
      swap_acceptance_rate=children[2],
      history_mean_fitness_per_island=children[3],
      history_max_fitness_per_island=children[4],
      history_ess_per_island=children[5],
      history_logZ_increment_per_island=children[6],
      history_meta_beta=children[7],
      history_num_accepted_swaps=children[8],
      history_num_attempted_swaps=children[9],
    )


jax.tree_util.register_pytree_node_class(ParallelReplicaSMCOutput)


@dataclass
class ParallelReplicaConfig:
  """Configuration for parallel replica SMC simulation."""

  sequence_length: int
  population_size_per_island: int
  n_islands: int
  exchange_frequency: int
  n_exchange_attempts_per_cycle: int
  ess_threshold_fraction: float
  meta_beta_schedule_type: str
  meta_beta_max_val: float
  meta_beta_schedule_rate: float
  mutation_rate: float
  sequence_type: Literal["protein", "nucleotide"]
  evolve_as: Literal["nucleotide", "protein"]


@dataclass(frozen=True)
class SMCStepConfig:
  sequence_length: int
  population_size_per_island: int
  mutation_rate: float
  fitness_evaluator: FitnessEvaluator
  sequence_type: Literal["protein", "nucleotide"]
  evolve_as: Literal["nucleotide", "protein"]
  ess_threshold_frac: float

  def tree_flatten(self):
    children = ()
    aux_data = self.__dict__
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(**aux_data)


@dataclass(frozen=True)
class ExchangeConfig:
  sequence_length: int
  population_size_per_island: int
  n_islands: int
  n_exchange_attempts: int
  fitness_evaluator: FitnessEvaluator
  sequence_type: Literal["protein", "nucleotide"]
  evolve_as: Literal["nucleotide", "protein"]

  def tree_flatten(self):
    children = ()
    aux_data = self.__dict__
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(**aux_data)


jax.tree_util.register_pytree_node_class(SMCStepConfig)
jax.tree_util.register_pytree_node_class(ExchangeConfig)


class IslandState(NamedTuple):
  key: PRNGKeyArray
  population: jnp.ndarray
  beta: jnp.ndarray
  logZ_estimate: jnp.ndarray


@partial(jit, static_argnames=("do_initial_mutation",))
def island_smc_step(
  island_state: IslandState,
  p_step: int,
  config: SMCStepConfig,
  do_initial_mutation: bool = False,
) -> tuple[IslandState, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
  """
  Performs one SMC step (mutation, weighting, UNCONDITIONAL resampling) for a single island.
  ESS is calculated for metrics only. ess_threshold_frac is not used for resampling decision.
  """
  key_mutate, key_fitness, key_resample, key_next_island = jax_random.split(island_state.key, 4)

  current_population = island_state.population

  mutated_population = jax.lax.cond(
    (p_step > 0) | do_initial_mutation,
    lambda x: dispatch_mutation(
      key_mutate, x, config.mutation_rate, config.sequence_type, config.evolve_as
    ),
    lambda x: x,
    current_population,
  )

  fitness_values, fitness_components = calculate_population_fitness(
    key_fitness, mutated_population, config.evolve_as, config.fitness_evaluator
  )

  log_potential_values = jnp.where(
    jnp.isneginf(fitness_values), -jnp.inf, island_state.beta * fitness_values
  )
  log_weights = log_potential_values

  # 3. UNCONDITIONAL Resampling
  # Note: This is unconditional resampling, not adaptive. TODO: allow for adaptive
  final_population_for_step, ess, normalized_weights = resample(
    key_resample, mutated_population, log_weights
  )

  finite_fitness_mask = jnp.isfinite(fitness_values)
  if isinstance(normalized_weights, jax.Array) and isinstance(finite_fitness_mask, jax.Array):
    weighted_finite_fitness = jnp.where(
      finite_fitness_mask, fitness_values * normalized_weights, 0.0
    )
    sum_weights_for_finite = jnp.sum(jnp.where(finite_fitness_mask, normalized_weights, 0.0))
  else:
    weighted_finite_fitness = jnp.array(0.0)
    sum_weights_for_finite = jnp.array(0.0)

  mean_fitness_this_step = jax.lax.cond(
    sum_weights_for_finite > 1e-9,
    lambda: jnp.sum(weighted_finite_fitness) / sum_weights_for_finite,
    lambda: jnp.nan,
  )

  max_fitness_this_step = jnp.max(jnp.where(jnp.isfinite(fitness_values), fitness_values, -jnp.inf))
  max_fitness_this_step = jnp.where(
    jnp.all(~jnp.isfinite(fitness_values)), jnp.nan, max_fitness_this_step
  )

  current_logZ_increment = calculate_logZ_increment(log_weights, config.population_size_per_island)

  updated_island_state = IslandState(
    key=key_next_island,
    population=final_population_for_step,
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
    island_smc_step,
    in_axes=(
      0,  # IslandState is a NamedTuple of arrays, so use 0 for all axes
      None,
      None,
      None,
    ),
    out_axes=(0, (0, 0, 0, 0)),
  ),
  static_argnames=("do_initial_mutation",),
)


@partial(jit)
def migrate(
  current_island_states_tuple: IslandState,
  mean_fitness_per_island: jnp.ndarray,
  meta_beta_current_value: jnp.ndarray,
  key_exchange: PRNGKeyArray,
  config: ExchangeConfig,
) -> tuple[IslandState, jax.Array]:
  """
  Perform replica exchange attempts. Acceptance is scaled by meta_beta_current_value.
  """
  all_population_stacked = current_island_states_tuple.population
  all_betas = current_island_states_tuple.beta
  num_swaps_accepted_total = jnp.array(0, dtype=jnp.int32)

  safe_mean_fitness = jnp.nan_to_num(mean_fitness_per_island, nan=0.0)
  min_fitness_val = jnp.min(safe_mean_fitness)
  shifted_fitness = safe_mean_fitness - min_fitness_val + 1e-9
  sum_shifted_fitness = jnp.sum(shifted_fitness)
  probs_idx1 = jax.lax.cond(
    (sum_shifted_fitness > 1e-9) & (config.n_islands > 0),
    lambda: shifted_fitness / sum_shifted_fitness,
    lambda: jnp.ones(config.n_islands, dtype=jnp.float32) / jnp.maximum(config.n_islands, 1.0),
  )

  def exchange_attempt_loop_body(attempt_idx, loop_state):
    key_attempt, current_population_state, current_accepted_swaps = loop_state
    (
      key_select_idx1,
      key_select_idx2,
      key_particle_choice1,
      key_particle_choice2,
      key_acceptance,
      key_next_attempt,
    ) = jax_random.split(key_attempt, 6)
    idx1 = jax_random.choice(key_select_idx1, jnp.arange(config.n_islands), p=probs_idx1)
    offset_for_idx2 = jax_random.randint(
      key_select_idx2, shape=(), minval=1, maxval=jnp.maximum(config.n_islands, 2)
    )
    idx2 = (idx1 + offset_for_idx2) % config.n_islands
    idx2 = jax.lax.cond(config.n_islands <= 1, lambda: idx1, lambda: idx2)
    island_a_population = current_population_state[idx1]
    island_b_population = current_population_state[idx2]
    beta_a = all_betas[idx1]
    beta_b = all_betas[idx2]
    sequence_idx_a = jax_random.randint(
      key_particle_choice1, shape=(), minval=0, maxval=config.population_size_per_island
    )
    sequence_idx_b = jax_random.randint(
      key_particle_choice2, shape=(), minval=0, maxval=config.population_size_per_island
    )
    config_a = island_a_population[sequence_idx_a]
    config_b = island_b_population[sequence_idx_b]

    # Calculate fitness for single sequences using new system
    fitness_a, _ = calculate_population_fitness(
      key_acceptance, jnp.expand_dims(config_a, 0), config.evolve_as, config.fitness_evaluator
    )
    fitness_b, _ = calculate_population_fitness(
      key_acceptance, jnp.expand_dims(config_b, 0), config.evolve_as, config.fitness_evaluator
    )
    fitness_a = fitness_a[0]  # Extract single value
    fitness_b = fitness_b[0]  # Extract single value

    # Modified acceptance probability with meta_beta_current_value
    log_acceptance_ratio = meta_beta_current_value * (beta_a - beta_b) * (fitness_b - fitness_a)
    log_acceptance_ratio = jnp.where(
      jnp.isinf(fitness_a) | jnp.isinf(fitness_b) | (idx1 == idx2), -jnp.inf, log_acceptance_ratio
    )
    accept = (
      jnp.log(jax_random.uniform(key_acceptance, shape=(), minval=1e-38, maxval=1.0))
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
    lambda: lax.fori_loop(
      0, config.n_exchange_attempts, exchange_attempt_loop_body, initial_loop_state
    )[1:],
    lambda: (all_population_stacked, num_swaps_accepted_total),
  )
  updated_island_states = current_island_states_tuple._replace(population=final_population_state)
  return updated_island_states, total_accepted_swaps


def run_parallel_replica_smc(
  master_sim_key: PRNGKeyArray,
  config: ParallelReplicaConfig,
  island_betas: list[float],
  n_smc_steps: int,
  fitness_evaluator: FitnessEvaluator,
  initial_population_generation_fn: Callable,
) -> ParallelReplicaSMCOutput:
  """
  Runs Parallel Replica SMC with the new fitness evaluation system.
  """
  print(
    f"Running Parallel Replica SMC (JAX) - Meta-Annealing: "
    f"{config.meta_beta_schedule_type.upper()}, MaxMetaBeta: "
    f"{config.meta_beta_max_val:.2f} - UNCONDITIONAL Resampling."
  )
  print(
    f"L={config.sequence_length}, Islands={config.n_islands}, "
    f"Population/Island={config.population_size_per_island}, Steps={n_smc_steps}"
  )

  key_init_islands, key_smc_loop, key_next_master = jax_random.split(master_sim_key, 3)

  # Generate initial population using the provided callable
  initial_population_array = initial_population_generation_fn(
    key_init_islands, config.n_islands, config.population_size_per_island
  )

  island_keys = jax_random.split(key_init_islands, config.n_islands)
  initial_island_states = IslandState(
    key=island_keys,
    population=initial_population_array,
    beta=jnp.array(island_betas, dtype=jnp.float32),
    logZ_estimate=jnp.zeros(config.n_islands, dtype=jnp.float32),
  )

  # Create sub-configs
  smc_config = SMCStepConfig(
    sequence_length=config.sequence_length,
    population_size_per_island=config.population_size_per_island,
    mutation_rate=config.mutation_rate,
    fitness_evaluator=fitness_evaluator,
    sequence_type=config.sequence_type,
    evolve_as=config.evolve_as,
    ess_threshold_frac=config.ess_threshold_fraction,
  )

  exchange_config = ExchangeConfig(
    sequence_length=config.sequence_length,
    population_size_per_island=config.population_size_per_island,
    n_islands=config.n_islands,
    n_exchange_attempts=config.n_exchange_attempts_per_cycle,
    fitness_evaluator=fitness_evaluator,
    sequence_type=config.sequence_type,
    evolve_as=config.evolve_as,
  )

  # Define the JAX-jitted SMC step function for jax.lax.scan
  @partial(
    jit,
    static_argnames=(
      "n_smc_steps_static",
      "exchange_frequency_static",
      "meta_beta_schedule_type_static",
      "meta_beta_max_val_static",
      "meta_beta_schedule_rate_static",
    ),
  )
  def _parallel_replica_scan_step(
    carry_state,
    scan_inputs_current_step,
    smc_config_static: SMCStepConfig,
    exchange_config_static: ExchangeConfig,
    n_smc_steps_static: int,
    exchange_frequency_static: int,
    meta_beta_schedule_type_static: str,
    meta_beta_max_val_static: float,
    meta_beta_schedule_rate_static: float,
  ):
    current_overall_state, current_smc_loop_key, total_swaps_accepted, total_swaps_attempted = (
      carry_state
    )
    p_step = scan_inputs_current_step

    key_step_smc, key_step_exchange, next_smc_loop_key = jax_random.split(current_smc_loop_key, 3)

    current_overall_state, island_metrics_this_step = vmapped_smc_step(
      current_overall_state,
      p_step,
      smc_config_static,
      False,
    )
    ess_p, mean_fit_p, max_fit_p, logZ_inc_p = island_metrics_this_step

    # Calculate current meta_beta based on schedule
    current_step_1_indexed = p_step + 1
    if meta_beta_schedule_type_static == "linear":
      current_meta_beta_py_scalar = linear_schedule(
        jnp.asarray(current_step_1_indexed, dtype=jnp.int32),
        jnp.asarray(n_smc_steps_static, dtype=jnp.int32),
        jnp.asarray(meta_beta_max_val_static, dtype=jnp.float32),
      )
    elif meta_beta_schedule_type_static == "exponential":
      current_meta_beta_py_scalar = exponential_schedule(
        jnp.asarray(current_step_1_indexed, dtype=jnp.int32),
        jnp.asarray(n_smc_steps_static, dtype=jnp.int32),
        jnp.asarray(meta_beta_max_val_static, dtype=jnp.float32),
        jnp.asarray(meta_beta_schedule_rate_static, dtype=jnp.float32),
      )
    elif meta_beta_schedule_type_static == "cosine":
      current_meta_beta_py_scalar = cosine_schedule(
        jnp.asarray(current_step_1_indexed, dtype=jnp.int32),
        jnp.asarray(n_smc_steps_static, dtype=jnp.int32),
        jnp.asarray(meta_beta_max_val_static, dtype=jnp.float32),
      )
    else:  # static
      current_meta_beta_py_scalar = jnp.array(meta_beta_max_val_static, dtype=jnp.float32)
    current_meta_beta_jax = jnp.float32(current_meta_beta_py_scalar)

    num_accepted_this_cycle = jnp.array(0, dtype=jnp.int32)
    total_swaps_attempted_this_cycle = jnp.array(0, dtype=jnp.int32)

    do_exchange = (p_step + 1) % exchange_frequency_static == 0

    current_overall_state, num_accepted_this_cycle = jax.lax.cond(
      do_exchange & (exchange_config_static.n_islands > 1),
      lambda: migrate(
        current_island_states_tuple=current_overall_state,
        mean_fitness_per_island=mean_fit_p,
        meta_beta_current_value=current_meta_beta_jax,
        key_exchange=key_step_exchange,
        config=exchange_config_static,
      ),
      lambda: (current_overall_state, jnp.array(0, dtype=jnp.int32)),
    )
    total_swaps_attempted_this_cycle = jax.lax.cond(
      do_exchange & (exchange_config_static.n_islands > 1),
      lambda: jnp.array(exchange_config_static.n_exchange_attempts, dtype=jnp.int32),
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

  scan_over_inputs = jnp.arange(n_smc_steps)

  final_carry_state, collected_outputs_scan = lax.scan(
    lambda carry, scan_in: _parallel_replica_scan_step(
      carry,
      scan_in,
      smc_config,
      exchange_config,
      n_smc_steps,
      config.exchange_frequency,
      config.meta_beta_schedule_type,
      config.meta_beta_max_val,
      config.meta_beta_schedule_rate,
    ),
    initial_carry,
    scan_over_inputs,
    length=n_smc_steps,
  )

  final_island_states, _, final_total_swaps_accepted, final_total_swaps_attempted = (
    final_carry_state
  )

  swap_acceptance_rate = jnp.where(
    final_total_swaps_attempted > 0,
    final_total_swaps_accepted / final_total_swaps_attempted,
    jnp.array(0.0, dtype=jnp.float32),
  )

  return ParallelReplicaSMCOutput(
    sequence_length=config.sequence_length,
    mutation_rate=config.mutation_rate,
    n_islands=config.n_islands,
    island_betas=jnp.array(island_betas),
    population_size_per_island=config.population_size_per_island,
    n_smc_steps=n_smc_steps,
    exchange_frequency=config.exchange_frequency,
    n_exchange_attempts_per_cycle=config.n_exchange_attempts_per_cycle,
    ess_threshold_fraction=config.ess_threshold_fraction,
    meta_beta_schedule_type=config.meta_beta_schedule_type,
    meta_beta_max_val=config.meta_beta_max_val,
    meta_beta_schedule_rate=config.meta_beta_schedule_rate,
    final_logZ_estimates_per_island=final_island_states.logZ_estimate,
    swap_acceptance_rate=swap_acceptance_rate,
    history_mean_fitness_per_island=collected_outputs_scan["mean_fitness_per_island"],
    history_max_fitness_per_island=collected_outputs_scan["max_fitness_per_island"],
    history_ess_per_island=collected_outputs_scan["ess_per_island"],
    history_logZ_increment_per_island=collected_outputs_scan["logZ_increment_per_island"],
    history_meta_beta=collected_outputs_scan["meta_beta"],
    history_num_accepted_swaps=collected_outputs_scan["num_accepted_swaps"],
    history_num_attempted_swaps=collected_outputs_scan["num_attempted_swaps"],
  )
