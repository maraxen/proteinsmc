"""Unit tests for HMC sampler data model.

Tests cover initialization and edge cases for HMC sampler config/model.
"""
import pytest
from proteinsmc.models.hmc import HMCConfig
from .conftest import basic_fitness_evaluator

@pytest.mark.parametrize("basic_fitness_evaluator", [basic_fitness_evaluator])
def test_hmc_config_initialization(basic_fitness_evaluator):
  """Test HMCConfig initialization with valid arguments.
  Args:
    None
  Returns:
    None
  Raises:
    AssertionError: If the config fields do not match expected values.
  Example:
    >>> test_hmc_config_initialization()
  """
  config = HMCConfig(
    num_samples=20,
    fitness_evaluator=basic_fitness_evaluator,
    step_size=0.01,
    num_integration_steps=10,
  )
  assert config.num_samples == 20
  assert config.step_size == 0.01
  assert config.num_integration_steps == 10
  assert config.fitness_evaluator == basic_fitness_evaluator
