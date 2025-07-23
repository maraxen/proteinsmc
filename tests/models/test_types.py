"""Unit tests for types data model.

Tests cover initialization and edge cases for types defined in types.py.
"""
import pytest
from proteinsmc.models import types as proteinsmc_types


def test_types_initialization():
  """Test types initialization with valid arguments.
  Args:
    None
  Returns:
    None
  Raises:
    AssertionError: If the types fields do not match expected values.
  Example:
    >>> test_types_initialization()
  """
  # Example: types.SequenceType
  seq_type = proteinsmc_types.SequenceType.PROTEIN
  assert seq_type == "protein"
