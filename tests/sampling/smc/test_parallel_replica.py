
import jax.numpy as jnp
import chex
import pytest
from jax import random

from proteinsmc.sampling.smc.parallel_replica import (
  ExchangeConfig,
  ParallelReplicaConfig,
  PRSMCStepConfig,
  prsmc_sampler,
)
from proteinsmc.utils.annealing_schedules import (
  AnnealingScheduleConfig,
  linear_schedule,
)
from proteinsmc.utils.fitness import FitnessEvaluator, FitnessFunction


def mock_fitness_fn(key, sequence, **kwargs) -> jnp.ndarray:
  """Mock fitness function that returns a single score per sequence."""
  return jnp.sum(sequence, axis=-1).astype(jnp.float32)


@pytest.fixture
def setup_sampler():
  """Provide a standard setup for the Parallel Replica SMC sampler test."""
  key = random.PRNGKey(0)
  seed_sequence = "MKY"
  n_islands = 4
  population_size_per_island = 8

  fitness_evaluator = FitnessEvaluator(
    fitness_functions=(
      FitnessFunction(
        func=mock_fitness_fn,
        input_type="protein",
        name="mock_fitness",
      ),
    ),
  )

  exchange_config = ExchangeConfig(
    population_size_per_island=population_size_per_island,
    n_islands=n_islands,
    n_exchange_attempts=5,
    fitness_evaluator=fitness_evaluator,
    exchange_frequency=10,
    sequence_type="protein",
    n_exchange_attempts_per_cycle=5,
    ess_threshold_fraction=0.5,
  )

  meta_beta_schedule_config = AnnealingScheduleConfig(
    schedule_fn=linear_schedule,
    beta_max=1.0,
    annealing_len=100,
  )

  step_config = PRSMCStepConfig(
    population_size_per_island=population_size_per_island,
    mutation_rate=0.1,
    fitness_evaluator=fitness_evaluator,
    sequence_type="protein",
    ess_threshold_frac=0.5,
    meta_beta_annealing_schedule=meta_beta_schedule_config,
    exchange_config=exchange_config,
  )

  config = ParallelReplicaConfig(
    seed_sequence=seed_sequence,
    population_size_per_island=population_size_per_island,
    n_islands=n_islands,
    n_states=20,
    generations=100,
    island_betas=[0.1] * n_islands,
    initial_diversity=0.1,
    fitness_evaluator=fitness_evaluator,
    step_config=step_config,
  )

  return key, config


def test_run_parallel_replica_smc_output(setup_sampler):
  """Tests that the Parallel Replica sampler runs and produces output with the correct shapes."""
  key, config = setup_sampler

  output = prsmc_sampler(key, config)

  assert output is not None
  chex.assert_shape(output.final_island_states.beta, (config.n_islands,))
  chex.assert_equal(output.final_island_states.population.shape[0], config.n_islands)
  chex.assert_shape(output.swap_acceptance_rate, ())
  chex.assert_shape(output.history_mean_fitness_per_island, (config.generations, config.n_islands))
  chex.assert_shape(output.history_ess_per_island, (config.generations, config.n_islands))
