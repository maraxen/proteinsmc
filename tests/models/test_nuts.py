"""Unit tests for NUTS sampler data model.

Tests cover initialization and edge cases for NUTS sampler config/model.
"""
import pytest
from proteinsmc.models import nuts, FitnessEvaluator


def test_nuts_config_initialization(fitness_evaluator_mock: FitnessEvaluator):
  """Test NUTSConfig initialization with valid arguments.
  Args:
    fitness_evaluator_mock: A mock fitness evaluator.
  Returns:
    None
  Raises:
    AssertionError: If the config fields do not match expected values.
  Example:
    >>> test_nuts_config_initialization(fitness_evaluator_mock)
  """
  config = nuts.NUTSConfig(
    num_samples=25,
    step_size=0.1,
    max_num_doublings=10,
    fitness_evaluator=fitness_evaluator_mock
  )
  assert config.num_samples == 25
  assert config.step_size == 0.1
  assert config.max_num_doublings == 10