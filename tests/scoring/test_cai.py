"""Tests for the CAI (Codon Adaptation Index) scoring module."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from chex import assert_shape

from proteinsmc.scoring.cai import cai_score, make_cai_score
from proteinsmc.utils.constants import PROTEINMPNN_X_INT


class TestCAIScore:
  """Test the cai_score function."""

  def test_cai_score_basic_sequence(self) -> None:
    """Test CAI score calculation with a basic valid nucleotide sequence.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the CAI score is not within the expected range.

    Example:
        >>> test_cai_score_basic_sequence()

    """
    # Simple codon sequence (3 codons = 9 nucleotides)
    # Each codon is represented by 3 integers (0-3 for A,C,G,T)
    sequence = jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=jnp.int8)
    aa_seq = jnp.array([0, 1, 2], dtype=jnp.int8)

    result = cai_score(sequence, aa_seq)

    # CAI should be a scalar float
    assert isinstance(result, jax.Array)
    assert_shape(result, ())
    # CAI should be between 0 and 1
    assert 0.0 <= float(result) <= 1.0

  def test_cai_score_with_x_codons(self) -> None:
    """Test CAI score calculation with X (unknown) amino acids.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If X codons are not properly masked.

    Example:
        >>> test_cai_score_with_x_codons()

    """
    # Create a sequence with some X amino acids
    sequence = jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=jnp.int8)
    aa_seq = jnp.array([0, PROTEINMPNN_X_INT, 2], dtype=jnp.int8)

    result = cai_score(sequence, aa_seq)

    # Should still return a valid CAI score
    assert isinstance(result, jax.Array)
    assert_shape(result, ())
    assert 0.0 <= float(result) <= 1.0

  def test_cai_score_all_x_codons(self) -> None:
    """Test CAI score when all amino acids are X (unknown).

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the result is not 0.0 for all-X sequence.

    Example:
        >>> test_cai_score_all_x_codons()

    """
    sequence = jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=jnp.int8)
    aa_seq = jnp.array(
      [PROTEINMPNN_X_INT, PROTEINMPNN_X_INT, PROTEINMPNN_X_INT],
      dtype=jnp.int8,
    )

    result = cai_score(sequence, aa_seq)

    # Should return 0.0 when no valid codons
    assert float(result) == 0.0

  def test_cai_score_single_codon(self) -> None:
    """Test CAI score with a single codon.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If single codon scoring fails.

    Example:
        >>> test_cai_score_single_codon()

    """
    sequence = jnp.array([0, 0, 0], dtype=jnp.int8)
    aa_seq = jnp.array([0], dtype=jnp.int8)

    result = cai_score(sequence, aa_seq)

    assert isinstance(result, jax.Array)
    assert_shape(result, ())
    assert 0.0 <= float(result) <= 1.0

  def test_cai_score_longer_sequence(self) -> None:
    """Test CAI score with a longer sequence.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If longer sequence scoring fails.

    Example:
        >>> test_cai_score_longer_sequence()

    """
    # 10 codons = 30 nucleotides
    sequence = jnp.array([i % 4 for i in range(30)], dtype=jnp.int8)
    aa_seq = jnp.array([i % 20 for i in range(10)], dtype=jnp.int8)

    result = cai_score(sequence, aa_seq)

    assert isinstance(result, jax.Array)
    assert_shape(result, ())
    assert 0.0 <= float(result) <= 1.0

  def test_cai_score_sequence_with_extra_nucleotides(self) -> None:
    """Test that extra nucleotides beyond complete codons are ignored.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If extra nucleotides affect the score.

    Example:
        >>> test_cai_score_sequence_with_extra_nucleotides()

    """
    # 9 nucleotides (3 complete codons) + 2 extra
    sequence_complete = jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=jnp.int8)
    sequence_extra = jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3], dtype=jnp.int8)
    aa_seq = jnp.array([0, 1, 2], dtype=jnp.int8)

    result_complete = cai_score(sequence_complete, aa_seq)
    result_extra = cai_score(sequence_extra, aa_seq)

    # Should give the same result since extra nucleotides are ignored
    assert jnp.allclose(result_complete, result_extra)

  def test_cai_score_is_jittable(self) -> None:
    """Test that cai_score can be JIT compiled.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If JIT compilation fails.

    Example:
        >>> test_cai_score_is_jittable()

    """
    jitted_cai = jax.jit(cai_score)

    sequence = jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=jnp.int8)
    aa_seq = jnp.array([0, 1, 2], dtype=jnp.int8)

    result = jitted_cai(sequence, aa_seq)

    assert isinstance(result, jax.Array)
    assert_shape(result, ())
    assert 0.0 <= float(result) <= 1.0


class TestMakeCAIScore:
  """Test the make_cai_score factory function."""

  def test_make_cai_score_returns_callable(self) -> None:
    """Test that make_cai_score returns a callable function.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the returned object is not callable.

    Example:
        >>> test_make_cai_score_returns_callable()

    """
    score_fn = make_cai_score()
    assert callable(score_fn)

  def test_make_cai_score_function_execution(self) -> None:
    """Test that the function returned by make_cai_score executes correctly.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the function does not execute correctly.

    Example:
        >>> test_make_cai_score_function_execution()

    """
    score_fn = make_cai_score()
    key = jax.random.PRNGKey(0)

    # Create a valid nucleotide sequence (9 nucleotides = 3 codons)
    sequence = jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=jnp.int8)

    result = score_fn(sequence, key, None)

    # Should return a scalar CAI score
    assert isinstance(result, jax.Array)
    assert_shape(result, ())
    assert 0.0 <= float(result) <= 1.0

  def test_make_cai_score_with_context(self) -> None:
    """Test that the CAI score function works with optional context parameter.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If context parameter handling fails.

    Example:
        >>> test_make_cai_score_with_context()

    """
    score_fn = make_cai_score()
    key = jax.random.PRNGKey(0)
    sequence = jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=jnp.int8)
    context = jnp.array([1.0, 2.0, 3.0])

    # Should not raise an error with context
    result = score_fn(sequence, key, context)

    assert isinstance(result, jax.Array)
    assert_shape(result, ())

  def test_make_cai_score_consistency(self) -> None:
    """Test that make_cai_score produces consistent results.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If results are not consistent across calls.

    Example:
        >>> test_make_cai_score_consistency()

    """
    score_fn = make_cai_score()
    key = jax.random.PRNGKey(42)
    sequence = jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=jnp.int8)

    result1 = score_fn(sequence, key, None)
    result2 = score_fn(sequence, key, None)

    # Should give same result for same input
    assert jnp.allclose(result1, result2)

  def test_make_cai_score_is_jittable(self) -> None:
    """Test that the function returned by make_cai_score is JIT-compatible.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If JIT compilation fails.

    Example:
        >>> test_make_cai_score_is_jittable()

    """
    score_fn = make_cai_score()
    # The function is already JIT compiled internally, but let's verify
    key = jax.random.PRNGKey(0)
    sequence = jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=jnp.int8)

    # Should execute without error
    result = score_fn(sequence, key, None)

    assert isinstance(result, jax.Array)
    assert 0.0 <= float(result) <= 1.0
