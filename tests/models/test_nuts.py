"""Unit tests for NUTS sampler data model.

Tests cover initialization and edge cases for NUTS sampler config/model.
"""
import pytest
from proteinsmc.models import nuts


def test_nuts_config_initialization():
  """Test NUTSConfig initialization with valid arguments.
  Args:
    None
  Returns:
    None
  Raises:
    AssertionError: If the config fields do not match expected values.
  Example:
    >>> test_nuts_config_initialization()
  """
  config = nuts.NUTSConfig(
    n_steps=25,
    max_tree_depth=10
  )
  assert config.n_steps == 25
  assert config.max_tree_depth == 10
