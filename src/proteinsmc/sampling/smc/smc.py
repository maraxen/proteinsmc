"""Core JIT-compiled logic for the SMC sampler."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import jit, vmap

from proteinsmc.models.mutation import MutationFn
from proteinsmc.models.smc import SMCConfig, SMCState
from proteinsmc.sampling.smc.resampling import resample
from proteinsmc.utils.initiate import generate_template_population
from proteinsmc.utils.metrics import calculate_logZ_increment, safe_weighted_mean, shannon_entropy

if TYPE_CHECKING:
  from jaxtyping import Int, PRNGKeyArray

  from proteinsmc.models.annealing import AnnealingFuncSignature
  from proteinsmc.models.fitness import StackedFitnessFn


def initialize_smc_state(
  config: SMCConfig,
  _fitness_function: StackedFitnessFn,
  key: PRNGKeyArray,
) -> SMCState:
  """Initialize the state for the SMC sampler."""
  key, subkey = jax.random.split(key)

  initial_population = generate_template_population(
    initial_sequence=config.seed_sequence,
    population_size=config.population_size,
    input_sequence_type=config.sequence_type,
    output_sequence_type=config.sequence_type,
  )

  return SMCState(
    population=initial_population,
    log_weights=jnp.full(config.population_size, -jnp.log(config.population_size)),
    log_likelihood=jnp.zeros(config.population_size),
    beta=0.0,
    key=subkey,
    step=0,
  )


@jit
def smc_step(
  state: SMCState,
  fitness_fn: StackedFitnessFn,
  mutation_fn: MutationFn,
) -> tuple[SMCState, dict]:
  """Perform one step of the SMC algorithm."""
  key_mutate, key_fitness, key_resample, key_next = jax.random.split(state.key, 4)

  mutated_population = vmap(
    mutation_fn,
    in_axes=(0, 0),
    out_axes=0,
  )(key_mutate, state.population)

  fitness_values, fitness_components = fitness_fn(
    mutated_population,
    key_fitness,
    None,
  )

  log_weights = jnp.where(jnp.isneginf(fitness_values), -jnp.inf, state.beta * fitness_values)

  logZ_increment = calculate_logZ_increment(log_weights, state.population.shape[0])  # noqa: N806
  resampled_population, ess, normalized_weights = resample(
    key_resample,
    mutated_population,
    log_weights,
  )

  valid_fitness_mask = jnp.isfinite(fitness_values)
  sum_valid_weights = jnp.sum(
    jnp.asarray(
      jnp.where(valid_fitness_mask, normalized_weights, 0.0),
      dtype=jnp.float32,
    ),
  )

  mean_combined_fitness = safe_weighted_mean(
    fitness_values,
    normalized_weights,
    valid_fitness_mask,
    sum_valid_weights,
  )

  max_combined_fitness = jnp.max(
    jnp.asarray(jnp.where(valid_fitness_mask, fitness_values, -jnp.inf), dtype=jnp.float32),
  )
  entropy = shannon_entropy(mutated_population)

  metrics = {
    "mean_combined_fitness": mean_combined_fitness,
    "max_combined_fitness": max_combined_fitness,
    "fitness_components": fitness_components,
    "ess": ess,
    "entropy": entropy,
    "beta": state.beta,
    "logZ_increment": logZ_increment,
  }

  next_state = SMCState(
    key=key_next,
    population=resampled_population,
    log_weights=normalized_weights,
    log_likelihood=fitness_values,
    beta=state.beta,
    step=state.step + 1,
  )

  return next_state, metrics


@partial(jit, static_argnames=("config", "log_prob_fn", "annealing_fn"))
def run_smc_loop(
  config: SMCConfig,
  initial_state: SMCState,
  fitness_fn: StackedFitnessFn,
  mutation_fn: MutationFn,
  annealing_fn: AnnealingFuncSignature,
) -> tuple[SMCState, dict]:
  """JIT-compiled SMC loop."""

  def scan_body(carry_state: SMCState, i: Int) -> tuple[SMCState, dict]:
    current_beta = annealing_fn(i, _context=None)  # type: ignore[call-arg]
    state_for_step = carry_state.replace(beta=current_beta)
    next_state, metrics = smc_step(state_for_step, config, fitness_fn, mutation_fn)
    return next_state, metrics

  final_state, collected_metrics = jax.lax.scan(
    scan_body,
    initial_state,
    jnp.arange(config.num_samples),
    length=config.num_samples,
  )

  return final_state, collected_metrics
