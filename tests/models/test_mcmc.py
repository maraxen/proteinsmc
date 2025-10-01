"""Unit tests for MCMC sampler data model.

Tests cover initialization and edge cases for MCMC sampler config/model.
"""
import pytest
from proteinsmc.models import mcmc, FitnessEvaluator


def test_mcmc_config_initialization(fitness_evaluator_mock: FitnessEvaluator):
  """Test MCMCConfig initialization with valid arguments.
  Args:
    fitness_evaluator_mock: A mock fitness evaluator.
  Returns:
    None
  Raises:
    AssertionError: If the config fields do not match expected values.
  Example:
    >>> test_mcmc_config_initialization(fitness_evaluator_mock)
  """
  config = mcmc.MCMCConfig(
    num_samples=15,
    step_size=0.05,
    fitness_evaluator=fitness_evaluator_mock,
  )
  assert config.num_samples == 15
  assert config.step_size == 0.05
  assert config.fitness_evaluator == fitness_evaluator_mock
