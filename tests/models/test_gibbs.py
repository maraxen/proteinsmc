"""Unit tests for Gibbs sampler data model.

Tests cover initialization and edge cases for Gibbs sampler config/model.
"""
import pytest
from proteinsmc.models import gibbs
from .conftest import basic_fitness_evaluator

@pytest.mark.parametrize("basic_fitness_evaluator", [basic_fitness_evaluator])
def test_gibbs_config_initialization(basic_fitness_evaluator):
  """Test GibbsConfig initialization with valid arguments.
  Args:
    None
  Returns:
    None
  Raises:
    AssertionError: If the config fields do not match expected values.
  Example:
    >>> test_gibbs_config_initialization()
  """
  config = gibbs.GibbsConfig(
    num_samples=10,
    mutation_rate=0.1,
    fitness_evaluator=basic_fitness_evaluator,
  )
  assert config.num_samples == 10
  assert config.mutation_rate == 0.1
  
