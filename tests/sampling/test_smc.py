import jax.numpy as jnp
import pytest
from jax import random

from proteinsmc.sampling.smc import (
  AnnealingScheduleConfig,
  SMCConfig,
  smc_sampler,
)
from proteinsmc.utils.annealing_schedules import linear_schedule
from proteinsmc.utils.fitness import FitnessEvaluator, FitnessFunction


# --- Mock Fitness Functions ---
def mock_protein_fitness_fn(key, seq, **kwargs):
  """Mock fitness function for protein sequences."""
  return jnp.sum(seq, axis=-1).astype(jnp.float32)


def mock_nucleotide_fitness_fn(key, seq, **kwargs):
  """Mock fitness function for nucleotide sequences."""
  return jnp.mean(seq, axis=-1).astype(jnp.float32)


# --- Fixture for Protein SMC ---
@pytest.fixture
def setup_smc_protein():
  """Provides a standard setup for a protein-based SMC sampler test."""
  key = random.PRNGKey(0)
  initial_sequence = "MKY"

  # Evaluator with only protein-based fitness functions
  fitness_evaluator = FitnessEvaluator(
    fitness_functions=[
      FitnessFunction(func=mock_protein_fitness_fn, input_type="protein", name="mock1", args={}),
      FitnessFunction(func=mock_protein_fitness_fn, input_type="protein", name="mock2", args={}),
    ]
  )

  smc_config = SMCConfig(
    sequence_length=len(initial_sequence),
    population_size=10,
    mutation_rate=0.1,
    sequence_type="protein",
    fitness_evaluator=fitness_evaluator,
    generations=5,
  )

  annealing_config = AnnealingScheduleConfig(
    schedule_fn=linear_schedule, beta_max=1.0, annealing_len=5
  )

  return key, smc_config, annealing_config, initial_sequence, 0.2


# --- Fixture for Nucleotide SMC ---
@pytest.fixture
def setup_smc_nucleotide():
  """Provides a standard setup for a nucleotide-based SMC sampler test."""
  key = random.PRNGKey(0)
  initial_sequence = "ATGAAATAC"

  # Evaluator with only nucleotide-based fitness functions
  fitness_evaluator = FitnessEvaluator(
    fitness_functions=[
      FitnessFunction(
        func=mock_nucleotide_fitness_fn, input_type="nucleotide", name="mock1", args={}
      ),
      FitnessFunction(
        func=mock_nucleotide_fitness_fn, input_type="nucleotide", name="mock2", args={}
      ),
    ]
  )

  smc_config = SMCConfig(
    sequence_length=len(initial_sequence),
    population_size=10,
    mutation_rate=0.1,
    sequence_type="nucleotide",
    fitness_evaluator=fitness_evaluator,
    generations=5,
  )

  annealing_config = AnnealingScheduleConfig(
    schedule_fn=linear_schedule, beta_max=1.0, annealing_len=5
  )

  return key, smc_config, annealing_config, initial_sequence, 0.2


# --- Corrected Tests ---


def test_smc_output_shapes(setup_smc_protein):
  """Tests that the SMC sampler output shapes are correct for a protein setup."""
  (
    key,
    smc_config,
    annealing_schedule_config,
    initial_sequence,
    diversification_ratio,
  ) = setup_smc_protein

  key_smc, key_init = random.split(key)

  output = smc_sampler(
    smc_config=smc_config,
    annealing_schedule_config=annealing_schedule_config,
    prng_key_smc_steps=key_smc,
    initial_population_key=key_init,
    diversification_ratio=diversification_ratio,
    initial_sequence=initial_sequence,
  )

  assert output.mean_combined_fitness_per_gen.shape == (smc_config.generations,)
  assert output.max_combined_fitness_per_gen.shape == (smc_config.generations,)
  assert output.entropy_per_gen.shape == (smc_config.generations,)


@pytest.mark.parametrize("fixture_name", ["setup_smc_protein", "setup_smc_nucleotide"])
def test_smc_sampler_runs(fixture_name, request):
  """Tests that the SMC sampler runs without error for both sequence types."""
  setup_data = request.getfixturevalue(fixture_name)
  (
    key,
    smc_config,
    annealing_schedule_config,
    initial_sequence,
    diversification_ratio,
  ) = setup_data

  key_smc, key_init = random.split(key)

  output = smc_sampler(
    smc_config=smc_config,
    annealing_schedule_config=annealing_schedule_config,
    prng_key_smc_steps=key_smc,
    initial_population_key=key_init,
    diversification_ratio=diversification_ratio,
    initial_sequence=initial_sequence,
  )

  assert output is not None
  assert output.final_logZhat is not None
