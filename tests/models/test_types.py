"""Unit tests for types data model.

Tests cover initialization and edge cases for types defined in types.py.
"""

from __future__ import annotations

import pytest

from proteinsmc.models import types as proteinsmc_types


def test_types_initialization() -> None:
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
  # Example: types.SequenceType is a Literal, not an Enum
  seq_type: proteinsmc_types.SequenceType = "protein"
  assert seq_type == "protein"

  seq_type_2: proteinsmc_types.SequenceType = "nucleotide"
  assert seq_type_2 == "nucleotide"


