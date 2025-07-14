
import jax.numpy as jnp
import chex
import pytest
from jax import random

from proteinsmc.sampling.smc import (
  SMCConfig,
  smc_sampler,
)
from proteinsmc.utils.annealing_schedules import AnnealingScheduleConfig, linear_schedule
from proteinsmc.utils.fitness import FitnessEvaluator, FitnessFunction


def mock_protein_fitness_fn(_key, seq, **kwargs):
  """Mock fitness function for protein sequences."""
  return jnp.sum(seq, axis=-1).astype(jnp.float32)


def mock_nucleotide_fitness_fn(_key, seq, **kwargs):
  """Mock fitness function for nucleotide sequences."""
  return jnp.mean(seq, axis=-1).astype(jnp.float32)


@pytest.fixture
def setup_smc_protein():
  """Provides a standard setup for a protein-based SMC sampler test."""
  key = random.PRNGKey(0)
  seed_sequence = "MKY"

  fitness_evaluator = FitnessEvaluator(
    fitness_functions=(
      FitnessFunction(func=mock_protein_fitness_fn, input_type="protein", name="mock1"),
      FitnessFunction(func=mock_protein_fitness_fn, input_type="protein", name="mock2"),
    )
  )

  annealing_config = AnnealingScheduleConfig(
    schedule_fn=linear_schedule,
    beta_max=1.0,
    annealing_len=5,
  )

  smc_config = SMCConfig(
    seed_sequence=seed_sequence,
    population_size=10,
    n_states=20,
    generations=5,
    mutation_rate=0.1,
    sequence_type="protein",
    annealing_schedule_config=annealing_config,
    fitness_evaluator=fitness_evaluator,
    diversification_ratio=0.2,
  )

  annealing_config = AnnealingScheduleConfig(
    schedule_fn=linear_schedule,
    beta_max=1.0,
    annealing_len=5,
  )

  return key, smc_config


@pytest.fixture
def setup_smc_nucleotide():
  """Provides a standard setup for a nucleotide-based SMC sampler test."""
  key = random.PRNGKey(0)
  seed_sequence = "ATGAAATAC"

  fitness_evaluator = FitnessEvaluator(
    fitness_functions=(
      FitnessFunction(
        func=mock_nucleotide_fitness_fn, input_type="nucleotide", name="mock1"
      ),
      FitnessFunction(
        func=mock_nucleotide_fitness_fn, input_type="nucleotide", name="mock2"
      ),
    )
  )

  annealing_config = AnnealingScheduleConfig(
    schedule_fn=linear_schedule,
    beta_max=1.0,
    annealing_len=5,
  )

  smc_config = SMCConfig(
    seed_sequence=seed_sequence,
    population_size=10,
    n_states=4,
    generations=5,
    mutation_rate=0.1,
    sequence_type="nucleotide",
    annealing_schedule_config=annealing_config,
    fitness_evaluator=fitness_evaluator,
    diversification_ratio=0.2,
  )

  return key, smc_config


def test_smc_output_shapes(setup_smc_protein):
  """Tests that the SMC sampler output shapes are correct for a protein setup."""
  (
    key,
    smc_config,
  ) = setup_smc_protein

  output = smc_sampler(
    key=key,
    config=smc_config,
  )

  chex.assert_shape(output.mean_combined_fitness_per_gen, (smc_config.generations,))
  chex.assert_shape(output.max_combined_fitness_per_gen, (smc_config.generations,))
  chex.assert_shape(output.entropy_per_gen, (smc_config.generations,))


@pytest.mark.parametrize("fixture_name", ["setup_smc_protein", "setup_smc_nucleotide"])
def test_smc_sampler_runs(fixture_name, request):
  """Tests that the SMC sampler runs without error for both sequence types."""
  setup_data = request.getfixturevalue(fixture_name)
  (key, smc_config) = setup_data

  output = smc_sampler(key=key, config=smc_config)

  assert output is not None
  assert output.final_logZhat is not None
