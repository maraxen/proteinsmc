"""Unit tests for NUTS sampler data model.

Tests cover initialization and edge cases for NUTS sampler config/model.
"""
import pytest
from proteinsmc.models import nuts, FitnessEvaluator


def test_nuts_config_initialization(basic_fitness_evaluator: FitnessEvaluator):
  """Test NUTSConfig initialization with valid arguments."""
  config = nuts.NUTSConfig(
    num_samples=25,
    max_num_doublings=10,
    fitness_evaluator=basic_fitness_evaluator,
  )
  assert config.num_samples == 25
  assert config.max_num_doublings == 10