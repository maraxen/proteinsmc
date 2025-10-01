"""Unit tests for SMCConfig data model.

Tests cover initialization, type checking, and edge cases for SMCConfig.
"""
import pytest
from proteinsmc.models import SMCConfig, FitnessEvaluator, MemoryConfig, AnnealingConfig, AutoTuningConfig


def test_smc_config_initialization(fitness_evaluator_mock: FitnessEvaluator):
  """Test SMCConfig initialization with valid arguments.
  Args:
    fitness_evaluator_mock: A mock fitness evaluator.
  Returns:
    None
  Raises:
    AssertionError: If the config fields do not match expected values.
  Example:
    >>> test_smc_config_initialization(fitness_evaluator_mock)
  """
  config = SMCConfig(
    prng_seed=42,
    seed_sequence="MKAF",
    num_samples=10,
    n_states=20,
    mutation_rate=0.1,
    diversification_ratio=0.5,
    sequence_type="protein",
    fitness_evaluator=fitness_evaluator_mock,
    memory_config=MemoryConfig(
      population_chunk_size=16,
      enable_chunked_vmap=True,
      device_memory_fraction=0.8,
      auto_tuning_config=AutoTuningConfig(),
    ),
    annealing_config=AnnealingConfig(
      annealing_fn="linear",
      beta_max=1.0,
      n_steps=10,
      kwargs={},
    ),
    population_size=16,
  )
  assert config.prng_seed == 42
  assert config.seed_sequence == "MKAF"
  assert config.num_samples == 10
  assert config.n_states == 20
  assert config.mutation_rate == 0.1
  assert config.diversification_ratio == 0.5
  assert config.sequence_type == "protein"
  assert config.population_size == 16