"""Unit tests for MutationConfig data model.

Tests cover initialization and edge cases for MutationConfig.
"""
import pytest
from proteinsmc.models import mutation


def test_mutation_config_initialization():
  """Test MutationConfig initialization with valid arguments.
  Args:
    None
  Returns:
    None
  Raises:
    AssertionError: If the config fields do not match expected values.
  Example:
    >>> test_mutation_config_initialization()
  """
  config = mutation.MutationConfig(
    mutation_rate=0.1,
    n_states=20
  )
  assert config.mutation_rate == 0.1
  assert config.n_states == 20
