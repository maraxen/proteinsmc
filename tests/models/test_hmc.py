"""Unit tests for HMC sampler data model.

Tests cover initialization and edge cases for HMC sampler config/model.
"""
import pytest
from proteinsmc.models.hmc import HMCConfig
from proteinsmc.models import FitnessEvaluator


def test_hmc_config_initialization(fitness_evaluator_mock: FitnessEvaluator):
  """Test HMCConfig initialization with valid arguments.
  Args:
    fitness_evaluator_mock: A mock fitness evaluator.
  Returns:
    None
  Raises:
    AssertionError: If the config fields do not match expected values.
  Example:
    >>> test_hmc_config_initialization(fitness_evaluator_mock)
  """
  config = HMCConfig(
    num_samples=20,
    fitness_evaluator=fitness_evaluator_mock,
    step_size=0.01,
    num_integration_steps=10,
  )
  assert config.num_samples == 20
  assert config.step_size == 0.01
  assert config.num_integration_steps == 10
  assert config.fitness_evaluator == fitness_evaluator_mock