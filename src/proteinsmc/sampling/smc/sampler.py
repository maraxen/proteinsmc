"""Implementation of the Sequential Monte Carlo (SMC) sampler for sequence design."""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import lax, random

from proteinsmc.models.smc import SMCCarryState, SMCConfig, SMCOutput
from proteinsmc.sampling.smc.step import smc_step
from proteinsmc.sampling.smc.validation import validate_smc_config
from proteinsmc.utils import (
  diversify_initial_sequences,
  generate_template_population,
  shannon_entropy,
)
from proteinsmc.utils.annealing import ANNEALING_REGISTRY

if TYPE_CHECKING:
  from jaxtyping import PRNGKeyArray


logger = getLogger(__name__)
logger.setLevel("INFO")


def smc_sampler(key: PRNGKeyArray, config: SMCConfig) -> SMCOutput:
  """Run a Sequential Monte Carlo simulation for sequence design."""
  validate_smc_config(config)
  initial_population = generate_template_population(
    initial_sequence=config.seed_sequence,
    population_size=config.population_size,
    input_sequence_type=config.sequence_type,
    output_sequence_type=config.sequence_type,
  )
  population_size = initial_population.shape[0]
  logger.info(
    "Running SMC (JAX) | Shape=%s | Schedule=%s | PopulationSize=%d | Steps=%d",
    initial_population.shape,
    config.annealing_schedule.schedule_fn,
    population_size,
    config.generations,
  )

  key, subkey = random.split(key)
  initial_population = diversify_initial_sequences(
    key=subkey,
    seed_sequences=initial_population,
    mutation_rate=config.diversification_ratio,
    sequence_type=config.sequence_type,
  )

  n_steps = jnp.array(config.annealing_schedule.n_steps, dtype=jnp.int32)
  beta_max = jnp.array(config.annealing_schedule.beta_max, dtype=jnp.float32)

  beta_schedule = lax.fori_loop(
    lower=0,
    upper=config.generations,
    body_fun=ANNEALING_REGISTRY[config.annealing_schedule.schedule_fn],
    init_val=(n_steps, beta_max),
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
    beta_current_step: jnp.ndarray,
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
