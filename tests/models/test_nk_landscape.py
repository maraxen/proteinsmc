"""Unit tests for NKLandscapeConfig data model.

Tests cover initialization and edge cases for NKLandscapeConfig.
"""
import pytest
from proteinsmc.models import nk_landscape


def test_nk_landscape_config_initialization():
  """Test NKLandscapeConfig initialization with valid arguments.
  Args:
    None
  Returns:
    None
  Raises:
    AssertionError: If the config fields do not match expected values.
  Example:
    >>> test_nk_landscape_config_initialization()
  """
  config = nk_landscape.NKLandscapeConfig(
    n=4,
    k=2
  )
  assert config.n == 4
  assert config.k == 2
