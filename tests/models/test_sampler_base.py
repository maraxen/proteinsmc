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
  assert isinstance(config.fitness_evaluator, FitnessEvaluator)
  assert isinstance(config.memory_config, MemoryConfig)


def test_base_sampler_config_initialization_with_jax_arrays(
  fitness_evaluator_mock: FitnessEvaluator,
):
  """Test BaseSamplerConfig initialization with jax.Array arguments."""
  import jax.numpy as jnp

  config = sampler_base.BaseSamplerConfig(
    prng_seed=123,
    sampler_type="smc",
    seed_sequence=jnp.array([0, 1, 2, 3]),
    num_samples=jnp.array(10),
    n_states=jnp.array(20),
    mutation_rate=jnp.array(0.05),
    diversification_ratio=jnp.array(0.2),
    sequence_type="protein",
    fitness_evaluator=fitness_evaluator_mock,
  )
  assert config.prng_seed == 123
  assert config.sampler_type == "smc"
  assert isinstance(config.seed_sequence, jnp.ndarray)
  assert isinstance(config.num_samples, jnp.ndarray)
  assert isinstance(config.n_states, jnp.ndarray)
  assert isinstance(config.mutation_rate, jnp.ndarray)
  assert isinstance(config.diversification_ratio, jnp.ndarray)
  assert config.sequence_type == "protein"
  assert isinstance(config.fitness_evaluator, FitnessEvaluator)
  assert isinstance(config.memory_config, MemoryConfig)
