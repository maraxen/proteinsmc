import jax.numpy as jnp
import pytest
from jax import random

from proteinsmc.sampling.parallel_replica import (
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


def mock_fitness_fn(key, sequence, **kwargs):
  """Mock fitness function that returns a single score per sequence."""
  return jnp.sum(sequence, axis=-1).astype(jnp.float32)


@pytest.fixture
def setup_sampler():
  """Provides a standard setup for the Parallel Replica SMC sampler test."""
  key = random.PRNGKey(0)
  sequence_length = 10
  n_islands = 4
  population_size_per_island = 8

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

  # 3. Create the main configuration object
  template_sequences = jnp.zeros(
    (population_size_per_island * n_islands, sequence_length), dtype=jnp.int32
  )

  config = ParallelReplicaConfig(
    template_sequences=template_sequences,
    population_size_per_island=population_size_per_island,
    n_islands=n_islands,
    n_states=20,  # For proteins
    generations=100,
    island_betas=jnp.linspace(0.1, 1.0, n_islands),
    initial_diversity=0.1,
    fitness_evaluator=fitness_evaluator,
    step_config=step_config,
  )

  return key, config


def test_run_parallel_replica_smc_output(setup_sampler):
  """Tests that the Parallel Replica SMC sampler runs and produces output with the correct shapes."""
  key, config = setup_sampler

  output = prsmc_sampler(key, config)

  assert output is not None
  assert output.final_island_states.beta.shape == (config.n_islands,)
  assert output.final_island_states.population.shape[0] == config.n_islands
  assert output.swap_acceptance_rate.shape == ()
  assert output.history_mean_fitness_per_island.shape == (
    config.generations,
    config.n_islands,
  )
  assert output.history_ess_per_island.shape == (
    config.generations,
    config.n_islands,
  )
