"""Unit tests for BaseSamplerConfig data model.

Tests cover initialization and edge cases for BaseSamplerConfig.
"""
import pytest
from proteinsmc.models import sampler_base, FitnessEvaluator, MemoryConfig


def test_base_sampler_config_initialization(fitness_evaluator_mock: FitnessEvaluator):
  """Test BaseSamplerConfig initialization with valid arguments.
  Args:
    fitness_evaluator_mock: A mock fitness evaluator.
  Returns:
    None
  Raises:
    AssertionError: If the config fields do not match expected values.
  Example:
    >>> test_base_sampler_config_initialization(fitness_evaluator_mock)
  """
  config = sampler_base.BaseSamplerConfig(
    prng_seed=123,
    sampler_type="smc",
    seed_sequence="ACDEFGHIKLMNPQRSTVWY",
    num_samples=10,
    n_states=20,
    mutation_rate=0.05,
    diversification_ratio=0.2,
    sequence_type="protein",
    fitness_evaluator=fitness_evaluator_mock,
  )
  assert config.prng_seed == 123
  assert config.sampler_type == "smc"
  assert config.seed_sequence == "ACDEFGHIKLMNPQRSTVWY"
  assert config.num_samples == 10
  assert config.n_states == 20
  assert config.mutation_rate == 0.05
  assert config.diversification_ratio == 0.2
  assert config.sequence_type == "protein"
  assert config.sequence_type == "protein"
  assert isinstance(config.fitness_evaluator, FitnessEvaluator)
  assert isinstance(config.memory_config, MemoryConfig)
