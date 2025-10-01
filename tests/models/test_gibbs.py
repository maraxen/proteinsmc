"""Unit tests for Gibbs sampler data model.

Tests cover initialization and edge cases for Gibbs sampler config/model.
"""
import pytest
from proteinsmc.models import gibbs, FitnessEvaluator


def test_gibbs_config_initialization(fitness_evaluator_mock: FitnessEvaluator):
  """Test GibbsConfig initialization with valid arguments.
  Args:
    fitness_evaluator_mock: A mock fitness evaluator.
  Returns:
    None
  Raises:
    AssertionError: If the config fields do not match expected values.
  Example:
    >>> test_gibbs_config_initialization(fitness_evaluator_mock)
  """
  config = gibbs.GibbsConfig(
    num_samples=10,
    mutation_rate=0.1,
    fitness_evaluator=fitness_evaluator_mock,
  )
  assert config.num_samples == 10
  assert config.mutation_rate == 0.1