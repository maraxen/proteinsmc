"""Tests for SMC validation utilities."""

import pytest

from proteinsmc.sampling.smc.data_structures import MemoryConfig, SMCConfig
from proteinsmc.sampling.smc.validation import validate_smc_config
from proteinsmc.utils import AnnealingScheduleConfig, FitnessEvaluator, FitnessFunction, linear_schedule


def mock_fitness_fn(key, sequence, **kwargs):
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
  invalid_config = SMCConfig(
    template_sequence="",
    population_size=valid_smc_config.population_size,
    n_states=valid_smc_config.n_states,
    generations=valid_smc_config.generations,
    mutation_rate=valid_smc_config.mutation_rate,
    diversification_ratio=valid_smc_config.diversification_ratio,
    sequence_type=valid_smc_config.sequence_type,
    annealing_schedule_config=valid_smc_config.annealing_schedule_config,
    fitness_evaluator=valid_smc_config.fitness_evaluator,
    memory_config=valid_smc_config.memory_config,
  )
  
  with pytest.raises(ValueError, match="Template sequence must be provided and cannot be empty"):
    validate_smc_config(invalid_config)


def test_validate_smc_config_invalid_sequence_type(valid_smc_config):
  """Test validation with invalid sequence type."""
  invalid_config = SMCConfig(
    template_sequence=valid_smc_config.template_sequence,
    population_size=valid_smc_config.population_size,
    n_states=valid_smc_config.n_states,
    generations=valid_smc_config.generations,
    mutation_rate=valid_smc_config.mutation_rate,
    diversification_ratio=valid_smc_config.diversification_ratio,
    sequence_type="invalid_type",  # type: ignore[arg-type]
    annealing_schedule_config=valid_smc_config.annealing_schedule_config,
    fitness_evaluator=valid_smc_config.fitness_evaluator,
    memory_config=valid_smc_config.memory_config,
  )
  
  with pytest.raises(ValueError, match="Invalid sequence type"):
    validate_smc_config(invalid_config)


def test_validate_smc_config_negative_population_size(valid_smc_config):
  """Test validation with negative population size."""
  invalid_config = SMCConfig(
    template_sequence=valid_smc_config.template_sequence,
    population_size=-10,
    n_states=valid_smc_config.n_states,
    generations=valid_smc_config.generations,
    mutation_rate=valid_smc_config.mutation_rate,
    diversification_ratio=valid_smc_config.diversification_ratio,
    sequence_type=valid_smc_config.sequence_type,
    annealing_schedule_config=valid_smc_config.annealing_schedule_config,
    fitness_evaluator=valid_smc_config.fitness_evaluator,
    memory_config=valid_smc_config.memory_config,
  )
  
  with pytest.raises(ValueError, match="Population size must be positive"):
    validate_smc_config(invalid_config)


def test_validate_smc_config_negative_generations(valid_smc_config):
  """Test validation with negative generations."""
  invalid_config = SMCConfig(
    template_sequence=valid_smc_config.template_sequence,
    population_size=valid_smc_config.population_size,
    n_states=valid_smc_config.n_states,
    generations=-5,
    mutation_rate=valid_smc_config.mutation_rate,
    diversification_ratio=valid_smc_config.diversification_ratio,
    sequence_type=valid_smc_config.sequence_type,
    annealing_schedule_config=valid_smc_config.annealing_schedule_config,
    fitness_evaluator=valid_smc_config.fitness_evaluator,
    memory_config=valid_smc_config.memory_config,
  )
  
  with pytest.raises(ValueError, match="Number of generations must be positive"):
    validate_smc_config(invalid_config)


def test_validate_smc_config_invalid_mutation_rate(valid_smc_config):
  """Test validation with invalid mutation rate."""
  invalid_config = SMCConfig(
    template_sequence=valid_smc_config.template_sequence,
    population_size=valid_smc_config.population_size,
    n_states=valid_smc_config.n_states,
    generations=valid_smc_config.generations,
    mutation_rate=1.5,  # > 1.0
    diversification_ratio=valid_smc_config.diversification_ratio,
    sequence_type=valid_smc_config.sequence_type,
    annealing_schedule_config=valid_smc_config.annealing_schedule_config,
    fitness_evaluator=valid_smc_config.fitness_evaluator,
    memory_config=valid_smc_config.memory_config,
  )
  
  with pytest.raises(ValueError, match="Mutation rate must be in the range"):
    validate_smc_config(invalid_config)


def test_validate_smc_config_invalid_diversification_ratio(valid_smc_config):
  """Test validation with invalid diversification ratio."""
  invalid_config = SMCConfig(
    template_sequence=valid_smc_config.template_sequence,
    population_size=valid_smc_config.population_size,
    n_states=valid_smc_config.n_states,
    generations=valid_smc_config.generations,
    mutation_rate=valid_smc_config.mutation_rate,
    diversification_ratio=1.2,  # > 1.0
    sequence_type=valid_smc_config.sequence_type,
    annealing_schedule_config=valid_smc_config.annealing_schedule_config,
    fitness_evaluator=valid_smc_config.fitness_evaluator,
    memory_config=valid_smc_config.memory_config,
  )
  
  with pytest.raises(ValueError, match="Diversification ratio must be"):
    validate_smc_config(invalid_config)


def test_validate_smc_config_invalid_fitness_evaluator(valid_smc_config):
  """Test validation with invalid fitness evaluator."""
  invalid_config = SMCConfig(
    template_sequence=valid_smc_config.template_sequence,
    population_size=valid_smc_config.population_size,
    n_states=valid_smc_config.n_states,
    generations=valid_smc_config.generations,
    mutation_rate=valid_smc_config.mutation_rate,
    diversification_ratio=valid_smc_config.diversification_ratio,
    sequence_type=valid_smc_config.sequence_type,
    annealing_schedule_config=valid_smc_config.annealing_schedule_config,
    fitness_evaluator="not_a_fitness_evaluator",  # type: ignore[arg-type]
    memory_config=valid_smc_config.memory_config,
  )
  
  with pytest.raises(TypeError, match="Expected fitness_evaluator to be an instance of FitnessEvaluator"):
    validate_smc_config(invalid_config)


def test_validate_smc_config_invalid_annealing_schedule(valid_smc_config):
  """Test validation with invalid annealing schedule config."""
  invalid_config = SMCConfig(
    template_sequence=valid_smc_config.template_sequence,
    population_size=valid_smc_config.population_size,
    n_states=valid_smc_config.n_states,
    generations=valid_smc_config.generations,
    mutation_rate=valid_smc_config.mutation_rate,
    diversification_ratio=valid_smc_config.diversification_ratio,
    sequence_type=valid_smc_config.sequence_type,
    annealing_schedule_config="not_an_annealing_config",  # type: ignore[arg-type]
    fitness_evaluator=valid_smc_config.fitness_evaluator,
    memory_config=valid_smc_config.memory_config,
  )
  
  with pytest.raises(TypeError, match="Expected annealing_schedule_config to be an instance"):
    validate_smc_config(invalid_config)
