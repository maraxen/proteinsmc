"""Implementation of the Sequential Monte Carlo (SMC) sampler for sequence design."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from logging import getLogger
from typing import TYPE_CHECKING, Literal

import jax
import jax.nn
import jax.numpy as jnp
from jax import jit, lax, random

if TYPE_CHECKING:
  from jaxtyping import PRNGKeyArray

  from proteinsmc.utils import (
    PopulationSequences,
    ScalarFloat,
    ScalarInt,
  )

from proteinsmc.utils import (
  AnnealingScheduleConfig,
  FitnessEvaluator,
  calculate_logZ_increment,
  calculate_population_fitness,
  dispatch_mutation,
  diversify_initial_sequences,
  generate_template_population,
  resample,
  shannon_entropy,
)

logger = getLogger(__name__)
logger.setLevel("INFO")


@dataclass(frozen=True)
class SMCConfig:
  """Configuration for the SMC sampler."""

  template_sequence: str
  population_size: int
  n_states: int
  generations: int
  mutation_rate: float
  diversification_ratio: float
  sequence_type: Literal["protein", "nucleotide"]
  annealing_schedule_config: AnnealingScheduleConfig
  fitness_evaluator: FitnessEvaluator

  def tree_flatten(self: SMCConfig) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility."""
    children: tuple = ()
    aux_data: dict = {k: v for k, v in self.__dict__.items() if k not in children}
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls: type[SMCConfig], aux_data: dict, _: tuple) -> SMCConfig:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(**aux_data)


@dataclass
class SMCCarryState:
  """State of the SMC sampler at a single step. Designed to be a JAX PyTree."""

  key: PRNGKeyArray
  population: PopulationSequences
  logZ_estimate: ScalarFloat  # noqa: N815
  beta: ScalarFloat
  step: ScalarInt = field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))

  def tree_flatten(self: SMCCarryState) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility."""
    children: tuple = (
      self.key,
      self.population,
      self.logZ_estimate,
      self.beta,
      self.step,
    )
    aux_data: dict = {}
    return children, aux_data

  @classmethod
  def tree_unflatten(
    cls: type[SMCCarryState],
    _aux_data: dict,
    children: tuple,
  ) -> SMCCarryState:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(*children)


@dataclass
class SMCOutput:
  """Output of the SMC sampler."""

  input_config: SMCConfig
  mean_combined_fitness_per_gen: jnp.ndarray
  max_combined_fitness_per_gen: jnp.ndarray
  entropy_per_gen: jnp.ndarray
  beta_per_gen: jnp.ndarray
  ess_per_gen: jnp.ndarray
  fitness_components_per_gen: jnp.ndarray
  final_logZhat: ScalarFloat  # noqa: N815
  final_amino_acid_entropy: ScalarFloat

  def tree_flatten(self: SMCOutput) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility."""
    children: tuple = (
      self.input_config,
      self.mean_combined_fitness_per_gen,
      self.max_combined_fitness_per_gen,
      self.entropy_per_gen,
      self.beta_per_gen,
      self.ess_per_gen,
      self.fitness_components_per_gen,
    )
    aux_data: dict = {
      "final_logZhat": self.final_logZhat,
      "final_amino_acid_entropy": self.final_amino_acid_entropy,
    }
    return (children, aux_data)

  @classmethod
  def tree_unflatten(
    cls: type[SMCOutput],
    aux_data: dict,
    children: tuple,
  ) -> SMCOutput:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(
      input_config=children[0],
      mean_combined_fitness_per_gen=children[1],
      max_combined_fitness_per_gen=children[2],
      entropy_per_gen=children[3],
      beta_per_gen=children[4],
      ess_per_gen=children[5],
      fitness_components_per_gen=children[6],
      **aux_data,
    )


jax.tree_util.register_pytree_node_class(SMCConfig)
jax.tree_util.register_pytree_node_class(SMCCarryState)
jax.tree_util.register_pytree_node_class(SMCOutput)


@partial(jit, static_argnames=("config",))
def smc_step(state: SMCCarryState, config: SMCConfig) -> tuple[SMCCarryState, dict]:
  """Perform one step of the SMC algorithm (mutate, weight, resample)."""
  key_mutate, key_fitness, key_resample, key_next = random.split(state.key, 4)

  mutated_population = dispatch_mutation(
    key_mutate,
    state.population,
    config.mutation_rate,
    config.sequence_type,
  )

  fitness_values, fitness_components = calculate_population_fitness(
    key_fitness,
    mutated_population,
    config.sequence_type,
    config.fitness_evaluator,
  )

  log_weights = jnp.where(jnp.isneginf(fitness_values), -jnp.inf, state.beta * fitness_values)

  logZ_increment = calculate_logZ_increment(log_weights, state.population.shape[0])  # noqa: N806
  resampled_population, ess, normalized_weights = resample(
    key_resample,
    mutated_population,
    log_weights,
  )
  if not isinstance(normalized_weights, jax.Array):
    msg = f"Expected normalized_weights to be a JAX array, got {type(normalized_weights)}"
    raise TypeError(
      msg,
    )
  valid_fitness_mask = jnp.isfinite(fitness_values)
  sum_valid_weights = jnp.sum(jnp.where(valid_fitness_mask, normalized_weights, 0.0))

  def safe_weighted_mean(
    metric: jax.Array,
    weights: jax.Array,
    valid_mask: jax.Array,
    sum_valid_w: ScalarFloat,
  ) -> jax.Array:
    if not isinstance(metric, jax.Array):
      msg = f"Expected metric to be a JAX array, got {type(metric)}"
      raise TypeError(msg)
    if not isinstance(weights, jax.Array):
      msg = f"Expected weights to be a JAX array, got {type(weights)}"
      raise TypeError(msg)
    eps = 1e-9
    output = jnp.where(
      sum_valid_w > eps,
      jnp.sum(jnp.where(valid_mask, metric * weights, 0.0)) / sum_valid_w,
      jnp.nan,
    )
    if not isinstance(output, jax.Array):
      msg = f"Expected output to be a JAX array, got {type(output)}"
      raise TypeError(msg)
    return output

  mean_combined_fitness = safe_weighted_mean(
    fitness_values,
    normalized_weights,
    valid_fitness_mask,
    sum_valid_weights,
  )
  if not isinstance(fitness_values, jax.Array):
    msg = f"Expected fitness_values to be a JAX array, got {type(fitness_values)}"
    raise TypeError(msg)
  max_combined_fitness = jnp.max(jnp.where(valid_fitness_mask, fitness_values, -jnp.inf))

  entropy = shannon_entropy(mutated_population)

  metrics = {
    "mean_combined_fitness": mean_combined_fitness,
    "max_combined_fitness": max_combined_fitness,
    "fitness_components": fitness_components,
    "ess": ess,
    "entropy": entropy,
    "beta": state.beta,
  }

  next_state = SMCCarryState(
    key=key_next,
    population=resampled_population,
    logZ_estimate=state.logZ_estimate + logZ_increment,
    beta=state.beta,
    step=state.step + 1,
  )
  return next_state, metrics


def _validate_smc_config(config: SMCConfig) -> None:
  """Validate the SMCConfig object to ensure all required fields are set correctly.

  Args:
    config: SMCConfig object to validate.


  Raises:
    ValueError or TypeError if any validation fails.

  """
  if not isinstance(config, SMCConfig):
    msg = f"Expected config to be an instance of SMCConfig, got {type(config)}"
    raise TypeError(msg)
  if config.template_sequence is None or config.template_sequence == "":
    msg = "Template sequence must be provided and cannot be empty."
    raise ValueError(msg)
  if config.sequence_type not in ["protein", "nucleotide"]:
    msg = f"Invalid sequence type '{config.sequence_type}'. Must be 'protein' or 'nucleotide'."
    raise ValueError(
      msg,
    )
  if config.n_states <= 0:
    msg = f"Number of states must be positive, got {config.n_states}."
    raise ValueError(msg)
  if config.generations <= 0:
    msg = f"Number of generations must be positive, got {config.generations}."
    raise ValueError(msg)
  if config.mutation_rate < 0 or config.mutation_rate > 1:
    msg = f"Mutation rate must be in the range [0, 1], got {config.mutation_rate}."
    raise ValueError(msg)
  if not isinstance(config.fitness_evaluator, FitnessEvaluator):
    msg = (
      f"Expected fitness_evaluator to be an instance of FitnessEvaluator, "
      f"got {type(config.fitness_evaluator)}"
    )
    raise TypeError(
      msg,
    )
  if not isinstance(config.annealing_schedule_config, AnnealingScheduleConfig):
    msg = (
      f"Expected annealing_schedule_config to be an instance of AnnealingScheduleConfig, "
      f"got {type(config.annealing_schedule_config)}"
    )
    raise TypeError(
      msg,
    )
  if not isinstance(config.diversification_ratio, float) or not (
    0 <= config.diversification_ratio <= 1
  ):
    msg = (
      f"Diversification ratio must be a float in the range [0, 1], got "
      f"{config.diversification_ratio}."
    )
    raise ValueError(
      msg,
    )


def smc_sampler(key: PRNGKeyArray, config: SMCConfig) -> SMCOutput:
  """Run a Sequential Monte Carlo simulation for sequence design."""
  _validate_smc_config(config)
  initial_population = generate_template_population(
    initial_sequence=config.template_sequence,
    population_size=config.population_size,
    input_sequence_type=config.sequence_type,
    output_sequence_type=config.sequence_type,
  )
  population_size = initial_population.shape[0]
  logger.info(
    "Running SMC (JAX) | Shape=%s | Schedule=%s | PopulationSize=%d | Steps=%d",
    initial_population.shape,
    config.annealing_schedule_config.schedule_fn.__name__,
    population_size,
    config.generations,
  )

  key, subkey = random.split(key)
  initial_population = diversify_initial_sequences(
    key=subkey,
    template_sequences=initial_population,
    mutation_rate=config.diversification_ratio,
    sequence_type=config.sequence_type,
  )

  annealing_len = jnp.array(config.annealing_schedule_config.annealing_len, dtype=jnp.int32)
  beta_max = jnp.array(config.annealing_schedule_config.beta_max, dtype=jnp.float32)

  beta_schedule = jnp.array(
    [
      config.annealing_schedule_config.schedule_fn(
        i + 1,
        annealing_len,
        beta_max,
      )
      for i in jnp.arange(config.generations)
    ],
    dtype=jnp.float32,
  )

  key, subkey = random.split(key)
  initial_state = SMCCarryState(
    key=subkey,
    population=initial_population,
    logZ_estimate=jnp.array(0.0, dtype=jnp.float32),
    beta=beta_schedule[0],
  )

  def scan_body(
    carry_state: SMCCarryState,
    beta_current_step: ScalarFloat,
  ) -> tuple[SMCCarryState, dict]:
    state_for_step = SMCCarryState(
      key=carry_state.key,
      population=carry_state.population,
      logZ_estimate=carry_state.logZ_estimate,
      beta=beta_current_step,
      step=carry_state.step,
    )
    next_state, metrics = smc_step(state_for_step, config)
    return next_state, metrics

  final_state, collected_metrics = lax.scan(
    scan_body,
    initial_state,
    beta_schedule,
    length=config.generations,
  )

  final_entropy = (
    shannon_entropy(final_state.population) if config.generations > 0 else jnp.array(jnp.nan)
  )

  logger.info("Finished JAX SMC. Final LogZhat=%.4f", float(final_state.logZ_estimate))

  return SMCOutput(
    input_config=config,
    mean_combined_fitness_per_gen=collected_metrics["mean_combined_fitness"],
    max_combined_fitness_per_gen=collected_metrics["max_combined_fitness"],
    entropy_per_gen=collected_metrics["entropy"],
    beta_per_gen=collected_metrics["beta"],
    ess_per_gen=collected_metrics["ess"],
    fitness_components_per_gen=collected_metrics["fitness_components"],
    final_logZhat=final_state.logZ_estimate,
    final_amino_acid_entropy=final_entropy,
  )
