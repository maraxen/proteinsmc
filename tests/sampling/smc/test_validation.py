"""Tests for SMC validation utilities."""

import dataclasses

import pytest

from proteinsmc.sampling.smc.data_structures import MemoryConfig, SMCConfig
from proteinsmc.sampling.smc.validation import validate_smc_config
from proteinsmc.utils import AnnealingScheduleConfig, FitnessEvaluator, FitnessFunction, linear_schedule


def mock_fitness_fn(_key, sequence, **_kwargs):
  """Mock fitness function for testing."""
  import jax.numpy as jnp
  return jnp.sum(sequence, axis=-1).astype(jnp.float32)


@pytest.fixture
def valid_fitness_evaluator():
  """Create a valid fitness evaluator for testing."""
  return FitnessEvaluator(
    fitness_functions=(
      FitnessFunction(
        func=mock_fitness_fn,
        input_type="protein",
        name="test_fitness",
      ),
    ),
  )


@pytest.fixture
def valid_annealing_config():
  """Create a valid annealing schedule config."""
  return AnnealingScheduleConfig(
    schedule_fn=linear_schedule,
    beta_max=1.0,
    annealing_len=50,
  )


@pytest.fixture
def valid_smc_config(valid_fitness_evaluator, valid_annealing_config):
  """Create a valid SMC config for testing."""
  return SMCConfig(
    template_sequence="MKYN",
    population_size=100,
    n_states=20,
    generations=50,
    mutation_rate=0.1,
    diversification_ratio=0.2,
    sequence_type="protein",
    annealing_schedule_config=valid_annealing_config,
    fitness_evaluator=valid_fitness_evaluator,
    memory_config=MemoryConfig(),
  )


def test_validate_smc_config_valid(valid_smc_config):
  """Test that a valid config passes validation."""
  # Should not raise any exceptions
  validate_smc_config(valid_smc_config)


def test_validate_smc_config_invalid_type():
  """Test validation with wrong config type."""
  with pytest.raises(TypeError, match="Expected config to be an instance of SMCConfig"):
    validate_smc_config("not_a_config")  # type: ignore[arg-type]
def test_validate_smc_config_empty_template_sequence(valid_smc_config):
  """Test validation with empty template sequence."""
  invalid_config = dataclasses.replace(valid_smc_config, template_sequence="")

  with pytest.raises(ValueError, match="Template sequence must be provided and cannot be empty"):
    validate_smc_config(invalid_config)


def test_validate_smc_config_invalid_sequence_type(valid_smc_config):
  """Test validation with invalid sequence type."""

  with pytest.raises(ValueError):
    dataclasses.replace(
    valid_smc_config, sequence_type="invalid_type"  # type: ignore[arg-type]
  )


def test_validate_smc_config_negative_population_size(valid_smc_config):
  """Test validation with negative population size."""
  with pytest.raises(ValueError, match="population_size must be positive."):
    dataclasses.replace(valid_smc_config, population_size=-10)



def test_validate_smc_config_negative_generations(valid_smc_config):
  """Test validation with negative generations."""
  with pytest.raises(ValueError):
    dataclasses.replace(valid_smc_config, generations=-5)


def test_validate_smc_config_invalid_mutation_rate(valid_smc_config):
  """Test validation with invalid mutation rate."""

  with pytest.raises(ValueError):
    dataclasses.replace(valid_smc_config, mutation_rate=1.5)


def test_validate_smc_config_invalid_diversification_ratio(valid_smc_config):
  """Test validation with invalid diversification ratio."""

  with pytest.raises(ValueError):
    dataclasses.replace(valid_smc_config, diversification_ratio=1.2)

def test_validate_smc_config_invalid_fitness_evaluator(valid_smc_config):
  """Test validation with invalid fitness evaluator."""
  invalid_config = dataclasses.replace(
    valid_smc_config,
    fitness_evaluator="not_a_fitness_evaluator",  # type: ignore[arg-type]
  )

  with pytest.raises(
    TypeError, match="Expected fitness_evaluator to be an instance of FitnessEvaluator"
  ):
    validate_smc_config(invalid_config)


def test_validate_smc_config_invalid_annealing_schedule(valid_smc_config):
  """Test validation with invalid annealing schedule config."""
  invalid_config = dataclasses.replace(
    valid_smc_config,
    annealing_schedule_config="not_an_annealing_config",  # type: ignore[arg-type]
  )

  with pytest.raises(TypeError, match="Expected annealing_schedule_config to be an instance"):
    validate_smc_config(invalid_config)
