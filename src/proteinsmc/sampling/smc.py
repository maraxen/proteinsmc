from dataclasses import dataclass, field
from functools import partial
from logging import getLogger
from typing import Literal

import jax
import jax.nn
import jax.numpy as jnp
from jax import jit, lax, random
from jaxtyping import PRNGKeyArray

from proteinsmc.utils.initiate import generate_template_population

from ..utils import (
  AnnealingScheduleConfig,
  FitnessEvaluator,
  PopulationSequences,
  ScalarFloat,
  ScalarInt,
  calculate_logZ_increment,
  calculate_population_fitness,
  diversify_initial_sequences,
  resample,
  shannon_entropy,
)
from ..utils.mutation import dispatch_mutation

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

  def tree_flatten(self):
    children = ()
    aux_data = {k: v for k, v in self.__dict__.items() if k not in children}
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(**aux_data)


@dataclass
class SMCCarryState:
  """State of the SMC sampler at a single step. Designed to be a JAX PyTree."""

  key: PRNGKeyArray
  population: PopulationSequences
  logZ_estimate: ScalarFloat
  beta: ScalarFloat
  step: ScalarInt = field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))

  def tree_flatten(self):
    children = (
      self.key,
      self.population,
      self.logZ_estimate,
      self.beta,
      self.step,
    )
    aux_data = {}
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*children)


@dataclass
class SMCOutput:
  """Output of the SMC sampler."""

  input_config: SMCConfig
  mean_combined_fitness_per_gen: jnp.ndarray
  max_combined_fitness_per_gen: jnp.ndarray
  mean_cai_per_gen: jnp.ndarray
  mean_mpnn_score_per_gen: jnp.ndarray
  entropy_per_gen: jnp.ndarray
  aa_entropy_per_gen: jnp.ndarray
  beta_per_gen: jnp.ndarray
  ess_per_gen: jnp.ndarray
  final_logZhat: float
  final_amino_acid_entropy: float

  def tree_flatten(self):
    children = (
      self.input_config,
      self.mean_combined_fitness_per_gen,
      self.max_combined_fitness_per_gen,
      self.mean_cai_per_gen,
      self.mean_mpnn_score_per_gen,
      self.entropy_per_gen,
      self.aa_entropy_per_gen,
      self.beta_per_gen,
      self.ess_per_gen,
    )
    aux_data = {
      "final_logZhat": self.final_logZhat,
      "final_amino_acid_entropy": self.final_amino_acid_entropy,
    }
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(
      input_config=children[0],
      mean_combined_fitness_per_gen=children[1],
      max_combined_fitness_per_gen=children[2],
      mean_cai_per_gen=children[3],
      mean_mpnn_score_per_gen=children[4],
      entropy_per_gen=children[5],
      aa_entropy_per_gen=children[6],
      beta_per_gen=children[7],
      ess_per_gen=children[8],
      **aux_data,
    )


jax.tree_util.register_pytree_node_class(SMCConfig)
jax.tree_util.register_pytree_node_class(SMCCarryState)
jax.tree_util.register_pytree_node_class(SMCOutput)


@partial(jit, static_argnames=("config",))
def smc_step(state: SMCCarryState, config: SMCConfig):
  """
  Performs one step of the SMC algorithm (mutate, weight, resample).
  """
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

  logZ_increment = calculate_logZ_increment(log_weights, state.population.shape[0])
  resampled_population, ess, normalized_weights = resample(
    key_resample, mutated_population, log_weights
  )
  if not isinstance(normalized_weights, jax.Array):
    raise TypeError(
      f"Expected normalized_weights to be a JAX array, got {type(normalized_weights)}"
    )
  valid_fitness_mask = jnp.isfinite(fitness_values)
  sum_valid_weights = jnp.sum(jnp.where(valid_fitness_mask, normalized_weights, 0.0))

  def safe_weighted_mean(metric, weights, valid_mask, sum_valid_w):
    if not isinstance(metric, jax.Array):
      raise TypeError(f"Expected metric to be a JAX array, got {type(metric)}")
    if not isinstance(weights, jax.Array):
      raise TypeError(f"Expected weights to be a JAX array, got {type(weights)}")
    return jnp.where(
      sum_valid_w > 1e-9,
      jnp.sum(jnp.where(valid_mask, metric * weights, 0.0)) / sum_valid_w,
      jnp.nan,
    )

  mean_combined_fitness = safe_weighted_mean(
    fitness_values, normalized_weights, valid_fitness_mask, sum_valid_weights
  )
  if not isinstance(fitness_values, jax.Array):
    raise TypeError(f"Expected fitness_values to be a JAX array, got {type(fitness_values)}")
  max_combined_fitness = jnp.max(jnp.where(valid_fitness_mask, fitness_values, -jnp.inf))

  cai_values = fitness_components.get("cai", jnp.zeros_like(fitness_values))
  valid_cai_mask = (cai_values > 0) & jnp.isfinite(cai_values)
  sum_valid_cai_weights = jnp.sum(jnp.where(valid_cai_mask, normalized_weights, 0.0))
  mean_cai = safe_weighted_mean(
    cai_values, normalized_weights, valid_cai_mask, sum_valid_cai_weights
  )

  mpnn_values = fitness_components.get("mpnn", jnp.zeros_like(fitness_values))
  valid_mpnn_mask = jnp.isfinite(mpnn_values)
  sum_valid_mpnn_weights = jnp.sum(jnp.where(valid_mpnn_mask, normalized_weights, 0.0))
  mean_mpnn_score = safe_weighted_mean(
    mpnn_values, normalized_weights, valid_mpnn_mask, sum_valid_mpnn_weights
  )

  entropy = shannon_entropy(mutated_population)
  aa_entropy = jnp.nan

  metrics = {
    "mean_combined_fitness": mean_combined_fitness,
    "max_combined_fitness": max_combined_fitness,
    "mean_cai": mean_cai,
    "mean_mpnn_score": mean_mpnn_score,
    "ess": ess,
    "entropy": entropy,
    "aa_entropy": aa_entropy,
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


def smc_sampler(key: PRNGKeyArray, config: SMCConfig) -> SMCOutput:
  """
  Runs a Sequential Monte Carlo simulation for sequence design.
  """
  if not isinstance(config, SMCConfig):
    raise TypeError(f"Expected config to be an instance of SMCConfig, got {type(config)}")
  if config.template_sequence is None or config.template_sequence == "":
    raise ValueError("Template sequence must be provided and cannot be empty.")
  if config.sequence_type not in ["protein", "nucleotide"]:
    raise ValueError(
      f"Invalid sequence type '{config.sequence_type}'. Must be 'protein' or 'nucleotide'."
    )
  if config.n_states <= 0:
    raise ValueError(f"Number of states must be positive, got {config.n_states}.")
  if config.generations <= 0:
    raise ValueError(f"Number of generations must be positive, got {config.generations}.")
  if config.mutation_rate < 0 or config.mutation_rate > 1:
    raise ValueError(f"Mutation rate must be in the range [0, 1], got {config.mutation_rate}.")
  if not isinstance(config.fitness_evaluator, FitnessEvaluator):
    raise TypeError(
      f"Expected fitness_evaluator to be an instance of FitnessEvaluator, got {type(config.fitness_evaluator)}"
    )
  if not isinstance(config.annealing_schedule_config, AnnealingScheduleConfig):
    raise TypeError(
      f"Expected annealing_schedule_config to be an instance of AnnealingScheduleConfig, "
      f"got {type(config.annealing_schedule_config)}"
    )
  if not isinstance(config.diversification_ratio, float) or not (
    0 <= config.diversification_ratio <= 1
  ):
    raise ValueError(
      f"Diversification ratio must be a float in the range [0, 1], got {config.diversification_ratio}."
    )
  initial_population = generate_template_population(
    initial_sequence=config.template_sequence,
    population_size=config.population_size,
    input_sequence_type=config.sequence_type,
    output_sequence_type=config.sequence_type,
  )
  population_size = initial_population.shape[0]
  logger.info(
    f"Running SMC (JAX) | Shape={initial_population.shape} | "
    f"Schedule={config.annealing_schedule_config.schedule_fn.__name__} | "
    f"PopulationSize={population_size} | Steps={config.generations}"
  )

  key, subkey = random.split(key)
  initial_population = diversify_initial_sequences(
    key=subkey,
    template_sequences=initial_population,
    mutation_rate=config.diversification_ratio,
    n_states=config.n_states,
    nucleotide=config.sequence_type == "nucleotide",
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

  def scan_body(carry_state, beta_current_step):
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
    scan_body, initial_state, beta_schedule, length=config.generations
  )

  final_aa_entropy = shannon_entropy(final_state.population) if config.generations > 0 else jnp.nan

  logger.info(f"Finished JAX SMC. Final LogZhat={float(final_state.logZ_estimate):.4f}")

  return SMCOutput(
    input_config=config,
    mean_combined_fitness_per_gen=collected_metrics["mean_combined_fitness"],
    max_combined_fitness_per_gen=collected_metrics["max_combined_fitness"],
    mean_cai_per_gen=collected_metrics["mean_cai"],
    mean_mpnn_score_per_gen=collected_metrics["mean_mpnn_score"],
    entropy_per_gen=collected_metrics["entropy"],
    aa_entropy_per_gen=collected_metrics["aa_entropy"],
    beta_per_gen=collected_metrics["beta"],
    ess_per_gen=collected_metrics["ess"],
    final_logZhat=float(final_state.logZ_estimate),
    final_amino_acid_entropy=float(final_aa_entropy),
  )
