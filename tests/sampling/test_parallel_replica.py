import jax.numpy as jnp
import pytest
from jax import random

from proteinsmc.sampling.parallel_replica import (
  ParallelReplicaConfig,
  run_parallel_replica_smc,
)
from proteinsmc.utils.fitness import FitnessEvaluator, FitnessFunction


# Mock fitness function for testing
def mock_fitness_fn(key, sequences, sequence_type):
  return jnp.sum(sequences, axis=-1), {}


# Mock initial population generation function
def mock_initial_population_fn(key, n_islands, population_size_per_island):
  return random.normal(key, (n_islands, population_size_per_island, 10))


@pytest.fixture
def setup_sampler():
  key = random.PRNGKey(0)
  config = ParallelReplicaConfig(
    sequence_length=10,
    population_size_per_island=8,
    n_islands=4,
    exchange_frequency=10,
    n_exchange_attempts_per_cycle=5,
    ess_threshold_fraction=0.5,
    meta_beta_schedule_type="linear",
    meta_beta_max_val=1.0,
    meta_beta_schedule_rate=0.9,
    mutation_rate=0.1,
    sequence_type="protein",
    evolve_as="protein",
  )
  island_betas = [1.0, 2.0, 4.0, 8.0]
  n_smc_steps = 100
  fitness_evaluator = FitnessEvaluator(
    fitness_functions=[
      FitnessFunction(
        func=mock_fitness_fn,
        input_type="protein",
        name="mock_fitness",
        args={},
      )
    ]
  )
  return key, config, island_betas, n_smc_steps, fitness_evaluator


def test_run_parallel_replica_smc_output(setup_sampler):
  (
    key,
    config,
    island_betas,
    n_smc_steps,
    fitness_evaluator,
  ) = setup_sampler

  output = run_parallel_replica_smc(
    key,
    config,
    island_betas,
    n_smc_steps,
    fitness_evaluator,
    mock_initial_population_fn,
  )

  assert output.final_logZ_estimates_per_island.shape == (config.n_islands,)
  assert output.swap_acceptance_rate.shape == ()
  assert output.history_mean_fitness_per_island.shape == (
    n_smc_steps,
    config.n_islands,
  )
