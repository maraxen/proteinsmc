"""Unit tests for TranslationConfig data model.

Tests cover initialization and edge cases for TranslationConfig.
"""
import pytest
from proteinsmc.models import translation


def test_translation_config_initialization():
  """Test TranslationConfig initialization with valid arguments.
  Args:
    None
  Returns:
    None
  Raises:
    AssertionError: If the config fields do not match expected values.
  Example:
    >>> test_translation_config_initialization()
  """
  config = translation.TranslationConfig(
    codon_table="standard",
    sequence_type="protein"
  )
  assert config.codon_table == "standard"
  assert config.sequence_type == "protein"
