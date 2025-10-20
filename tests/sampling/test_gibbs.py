"""Tests for Gibbs sampling algorithm."""

from __future__ import annotations

import pytest

from proteinsmc.sampling.gibbs import make_gibbs_update_fns


class TestMakeGibbsUpdateFns:
  """Test the make_gibbs_update_fns factory function."""

  def test_creates_correct_number_of_functions(self) -> None:
    """Test that correct number of update functions are created.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If wrong number of functions created.

    Example:
        >>> test_creates_correct_number_of_functions()

    """
    sequence_length = 5
    n_states = 4

    update_fns = make_gibbs_update_fns(sequence_length, n_states)

    assert len(update_fns) == sequence_length
    for fn in update_fns:
      assert callable(fn)

  def test_invalid_sequence_length(self) -> None:
    """Test with invalid sequence length.

    Args:
        None

    Returns:
        None

    Raises:
        ValueError: If sequence length is invalid.

    Example:
        >>> test_invalid_sequence_length()

    """
    with pytest.raises(ValueError, match="sequence_length and n_states must be positive"):
      make_gibbs_update_fns(sequence_length=0, n_states=4)

  def test_invalid_n_states(self) -> None:
    """Test with invalid n_states.

    Args:
        None

    Returns:
        None

    Raises:
        ValueError: If n_states is invalid.

    Example:
        >>> test_invalid_sequence_length()

    """
    with pytest.raises(ValueError, match="sequence_length and n_states must be positive"):
      make_gibbs_update_fns(sequence_length=5, n_states=0)

  def test_negative_sequence_length(self) -> None:
    """Test with negative sequence length.

    Args:
        None

    Returns:
        None

    Raises:
        ValueError: If sequence length is negative.

    Example:
        >>> test_negative_sequence_length()

    """
    with pytest.raises(ValueError, match="sequence_length and n_states must be positive"):
      make_gibbs_update_fns(sequence_length=-1, n_states=4)

  def test_different_n_states(self) -> None:
    """Test with different values of n_states.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If function creation fails.

    Example:
        >>> test_different_n_states()

    """
    sequence_length = 10

    for n_states in [4, 20, 64]:
      update_fns = make_gibbs_update_fns(sequence_length, n_states)
      assert len(update_fns) == sequence_length

  def test_different_sequence_lengths(self) -> None:
    """Test with different sequence lengths.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If function creation fails.

    Example:
        >>> test_different_sequence_lengths()

    """
    n_states = 4

    for length in [1, 5, 20, 100]:
      update_fns = make_gibbs_update_fns(length, n_states)
      assert len(update_fns) == length
      for fn in update_fns:
        assert callable(fn)
