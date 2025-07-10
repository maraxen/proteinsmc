"""Implementation of Parallel Replica inspired Sequential Monte Carlo (PRSMC) sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
from jax import jit, lax, vmap

if TYPE_CHECKING:
  from jaxtyping import Float, Int, PRNGKeyArray

  from proteinsmc.utils import (
    AnnealingScheduleConfig,
    FitnessEvaluator,
    IslandFloats,
    PopulationSequences,
  )

from proteinsmc.utils import (
  calculate_logZ_increment,
  calculate_population_fitness,
  dispatch_mutation,
  diversify_initial_sequences,
  generate_template_population,
  resample,
)


@dataclass(frozen=True)
class ExchangeConfig:
  """Configuration for parallel replica exchange."""

  population_size_per_island: int
  n_islands: int
  n_exchange_attempts: int
  fitness_evaluator: FitnessEvaluator
  exchange_frequency: float
  sequence_type: Literal["protein", "nucleotide"]
  n_exchange_attempts_per_cycle: int
  ess_threshold_fraction: float

  def tree_flatten(self: ExchangeConfig) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility."""
    children: tuple = ()
    aux_data: dict = self.__dict__
    return (children, aux_data)

  @classmethod
  def tree_unflatten(
    cls: type[ExchangeConfig],
    aux_data: dict,
    _children: tuple,
  ) -> ExchangeConfig:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(**aux_data)


@dataclass(frozen=True)
class PRSMCStepConfig:
  """Configuration for a single step in the PRSMC algorithm."""

  population_size_per_island: int
  mutation_rate: float
  fitness_evaluator: FitnessEvaluator
  sequence_type: Literal["protein", "nucleotide"]
  ess_threshold_frac: float
  meta_beta_annealing_schedule: AnnealingScheduleConfig
  exchange_config: ExchangeConfig

  def tree_flatten(self: PRSMCStepConfig) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility."""
    children: tuple = ()
    aux_data: dict = self.__dict__
    return (children, aux_data)

  @classmethod
  def tree_unflatten(
    cls: type[PRSMCStepConfig],
    aux_data: dict,
    _children: tuple,
  ) -> PRSMCStepConfig:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(**aux_data)


@dataclass(frozen=True)
class ParallelReplicaConfig:
  """Configuration for parallel replica SMC simulation."""

  template_sequence: str
  population_size_per_island: int
  n_islands: int
  n_states: int
  generations: int
  island_betas: list[float]
  initial_diversity: float
  fitness_evaluator: FitnessEvaluator
  step_config: PRSMCStepConfig

  def tree_flatten(self: ParallelReplicaConfig) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility."""
    _children: tuple = ()
    aux_data: dict = self.__dict__
    return (_children, aux_data)

  @classmethod
  def tree_unflatten(
    cls: type[ParallelReplicaConfig],
    aux_data: dict,
    _children: tuple,
  ) -> ParallelReplicaConfig:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(**aux_data)


@dataclass
class IslandState:
  """State of a single island in the PRSMC algorithm."""

  key: PRNGKeyArray
  population: PopulationSequences
  beta: Float
  logZ_estimate: Float  # noqa: N815
  ess: Float
  mean_fitness: Float
  step: Int = field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))

  def tree_flatten(self: IslandState) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility."""
    children: tuple = (
      self.key,
      self.population,
      self.beta,
      self.logZ_estimate,
      self.ess,
      self.mean_fitness,
      self.step,
    )
    aux_data: dict = {}
    return children, aux_data

  @classmethod
  def tree_unflatten(cls: type[IslandState], _aux_data: dict, children: tuple) -> IslandState:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(*children)


@dataclass
class PRSMCCarryState:
  """Carry state for the PRSMC algorithm, containing overall state and PRNG key."""

  current_overall_state: IslandState
  prng_key: PRNGKeyArray
  total_swaps_attempted: Int = field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))
  total_swaps_accepted: Int = field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))

  def tree_flatten(self: PRSMCCarryState) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility."""
    children: tuple = (
      self.current_overall_state,
      self.prng_key,
      self.total_swaps_attempted,
      self.total_swaps_accepted,
    )
    aux_data: dict = {}
    return children, aux_data

  @classmethod
  def tree_unflatten(
    cls: type[PRSMCCarryState],
    _aux_data: dict,
    children: tuple,
  ) -> PRSMCCarryState:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(*children)


@dataclass
class ParallelReplicaSMCOutput:
  """Output of the Parallel Replica SMC algorithm."""

  input_config: ParallelReplicaConfig
  final_island_states: IslandState
  swap_acceptance_rate: IslandFloats
  history_mean_fitness_per_island: IslandFloats
  history_max_fitness_per_island: IslandFloats
  history_ess_per_island: IslandFloats
  history_logZ_increment_per_island: IslandFloats  # noqa: N815
  history_meta_beta: IslandFloats
  history_num_accepted_swaps: IslandFloats
  history_num_attempted_swaps: IslandFloats

  def tree_flatten(self: ParallelReplicaSMCOutput) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility."""
    children: tuple = (
      self.input_config,
      self.final_island_states,
      self.swap_acceptance_rate,
      self.history_mean_fitness_per_island,
      self.history_max_fitness_per_island,
      self.history_ess_per_island,
      self.history_logZ_increment_per_island,
      self.history_meta_beta,
      self.history_num_accepted_swaps,
      self.history_num_attempted_swaps,
    )
    aux_data: dict = {}
    return children, aux_data

  @classmethod
  def tree_unflatten(
    cls: type[ParallelReplicaSMCOutput],
    _aux_data: dict,
    children: tuple,
  ) -> ParallelReplicaSMCOutput:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(*children)


jax.tree_util.register_pytree_node_class(ExchangeConfig)
jax.tree_util.register_pytree_node_class(PRSMCStepConfig)
jax.tree_util.register_pytree_node_class(ParallelReplicaConfig)
jax.tree_util.register_pytree_node_class(IslandState)
jax.tree_util.register_pytree_node_class(PRSMCCarryState)
jax.tree_util.register_pytree_node_class(ParallelReplicaSMCOutput)


@partial(jit, static_argnames=("config",))
def island_smc_step(
  island_state: IslandState,
  config: PRSMCStepConfig,
) -> IslandState:
  """Perform one SMC step (mutation, weighting, UNCONDITIONAL resampling) for a single island.

  ESS is calculated for metrics only currently.
  ess_threshold_frac is not used for resampling decision.

  Args:
    island_state (IslandState): Current state of the island.
    config (PRSMCStepConfig): Configuration for the SMC step.

  Returns:
    IslandState: Updated state of the island.

  """
  key_mutate, key_fitness, key_resample, key_next_island = jax.random.split(island_state.key, 4)
  current_population = island_state.population
  p_step = island_state.step

  mutated_population = jax.lax.cond(
    (p_step > 0),
    lambda x: dispatch_mutation(key_mutate, x, config.mutation_rate, config.sequence_type).astype(
      x.dtype,
    ),
    lambda x: x,
    current_population,
  )

  fitness_values, fitness_components = calculate_population_fitness(
    key_fitness,
    mutated_population,
    config.sequence_type,
    config.fitness_evaluator,
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


@partial(jit, static_argnames=("config",))
def island_smc_step_with_metrics(
  island_state: IslandState,
  config: PRSMCStepConfig,
) -> tuple[IslandState, tuple[Float, Float, Float, Float]]:
  """Perform a single step of the island SMC algorithm with metrics.

  Args:
    island_state (IslandState): Current state of the island.
    config (PRSMCStepConfig): Configuration for the SMC step.

  Returns:
    tuple[IslandState, tuple[Float, Float, Float, Float]]:
      Updated island state and a tuple containing:
        - Effective Sample Size (ESS)
        - Mean fitness of the population
        - Maximum fitness of the population
        - LogZ increment for the step

  """
  updated_island_state = island_smc_step(island_state, config)
  fitness_values, _ = calculate_population_fitness(
    updated_island_state.key,
    updated_island_state.population,
    config.sequence_type,
    config.fitness_evaluator,
  )
  ess = updated_island_state.ess
  mean_fitness = updated_island_state.mean_fitness
  if not isinstance(fitness_values, jax.Array):
    msg = f"Expected fitness_values to be a jax.Array, got {type(fitness_values)}"
    raise TypeError(msg)
  max_fitness = jnp.max(jnp.where(jnp.isfinite(fitness_values), fitness_values, -jnp.inf))
  logZ_increment = updated_island_state.logZ_estimate - island_state.logZ_estimate  # noqa: N806
  return updated_island_state, (ess, mean_fitness, max_fitness, logZ_increment)


vmapped_smc_step = jit(
  vmap(
    island_smc_step_with_metrics,
    in_axes=(0, None),
    out_axes=(0, (0, 0, 0, 0)),
  ),
  static_argnames=("config",),
)


@partial(jit, static_argnames=("config",))
def migrate(
  island_states: IslandState,
  meta_beta_current_value: Float,
  key_exchange: PRNGKeyArray,
  config: ExchangeConfig,
) -> tuple[IslandState, Int]:
  """Perform replica exchange attempts. Acceptance is scaled by meta_beta_current_value.

  Args:
    island_states (IslandState): Current state of the islands.
    meta_beta_current_value (Float): Current value of the meta beta parameter.
    key_exchange (PRNGKeyArray): JAX PRNG key for random operations.
    config (ExchangeConfig): Configuration for the exchange process.

  Returns:
    tuple[IslandState, Int]: Updated island states and total number of accepted swaps.

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

    fitness_a, _ = calculate_population_fitness(
      key_acceptance,
      jnp.expand_dims(config_a, 0),
      config.sequence_type,
      config.fitness_evaluator,
    )
    fitness_b, _ = calculate_population_fitness(
      key_acceptance,
      jnp.expand_dims(config_b, 0),
      config.sequence_type,
      config.fitness_evaluator,
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


def prsmc_sampler(
  key: PRNGKeyArray,
  config: ParallelReplicaConfig,
) -> ParallelReplicaSMCOutput:
  """Run Parallel Replica inspired SMC with the new fitness evaluation system."""
  key_init_islands, key_smc_loop = jax.random.split(key)
  initial_population = generate_template_population(
    initial_sequence=config.template_sequence,
    population_size=config.population_size_per_island * config.n_islands,
    input_sequence_type=config.step_config.sequence_type,
    output_sequence_type=config.step_config.sequence_type,
  )
  initial_population = diversify_initial_sequences(
    key=key_init_islands,
    template_sequences=initial_population,
    mutation_rate=config.initial_diversity,
    sequence_type=config.step_config.sequence_type,
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

  @partial(jit, static_argnames=("step_config",))
  def _parallel_replica_scan_step(
    carry_state: PRSMCCarryState,
    step_idx: Int,
    step_config: PRSMCStepConfig,
  ) -> tuple[PRSMCCarryState, dict]:
    key_step, next_smc_loop_key = jax.random.split(carry_state.prng_key)

    current_overall_state, island_metrics = vmapped_smc_step(
      carry_state.current_overall_state,
      step_config,
    )
    ess_p, mean_fit_p, max_fit_p, logZ_inc_p = island_metrics  # noqa: N806

    meta_annealing_schedule = step_config.meta_beta_annealing_schedule
    annealing_len = jnp.array(meta_annealing_schedule.annealing_len, dtype=jnp.int32)
    beta_max = jnp.array(meta_annealing_schedule.beta_max, dtype=jnp.float32)
    current_meta_beta = lax.cond(
      step_idx >= meta_annealing_schedule.annealing_len,
      lambda: jnp.array(meta_annealing_schedule.beta_max, dtype=jnp.float32),
      lambda: meta_annealing_schedule.schedule_fn(
        step_idx + 1,
        annealing_len,
        beta_max,
        *meta_annealing_schedule.schedule_args,
      ).astype(jnp.float32),
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
      ),
      lambda: (current_overall_state, jnp.array(0, dtype=jnp.int32)),
    )

    total_swaps_attempted_this_cycle = lax.cond(
      do_exchange & (exchange_config.n_islands > 1),
      lambda: jnp.array(exchange_config.n_exchange_attempts, dtype=jnp.int32),
      lambda: jnp.array(0, dtype=jnp.int32),
    )

    next_carry_state = PRSMCCarryState(
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

  initial_carry = PRSMCCarryState(
    initial_island_states,
    key_smc_loop,
    jnp.array(0, dtype=jnp.int32),
    jnp.array(0, dtype=jnp.int32),
  )
  step_config = config.step_config

  final_carry_state, collected_outputs_scan = lax.scan(
    lambda carry, scan_input: _parallel_replica_scan_step(
      carry,
      scan_input,
      step_config,
    ),
    initial_carry,
    jnp.arange(config.generations, dtype=jnp.int32),
    length=config.generations,
  )

  final_island_states = final_carry_state.current_overall_state
  final_total_swaps_accepted = final_carry_state.total_swaps_accepted
  final_total_swaps_attempted = final_carry_state.total_swaps_attempted

  swap_acceptance_rate = jnp.where(
    final_total_swaps_attempted > 0,
    final_total_swaps_accepted / final_total_swaps_attempted,
    jnp.array(0.0, dtype=jnp.float32),
  )

  if not isinstance(swap_acceptance_rate, jax.Array):
    msg = f"Expected swap_acceptance_rate to be a jax.Array, got {type(swap_acceptance_rate)}"
    raise TypeError(msg)

  return ParallelReplicaSMCOutput(
    input_config=config,
    final_island_states=final_island_states,
    swap_acceptance_rate=swap_acceptance_rate,
    history_mean_fitness_per_island=collected_outputs_scan["mean_fitness_per_island"],
    history_max_fitness_per_island=collected_outputs_scan["max_fitness_per_island"],
    history_ess_per_island=collected_outputs_scan["ess_per_island"],
    history_logZ_increment_per_island=collected_outputs_scan["logZ_increment_per_island"],
    history_meta_beta=collected_outputs_scan["meta_beta"],
    history_num_accepted_swaps=collected_outputs_scan["num_accepted_swaps"],
    history_num_attempted_swaps=collected_outputs_scan["num_attempted_swaps"],
  )
