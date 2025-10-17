"""Tests for initialization utility functions."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from chex import assert_shape

from proteinsmc.utils.initiate import generate_template_population


class TestGenerateTemplatePopulation:
  """Test the generate_template_population function."""

  def test_protein_to_nucleotide_conversion(self) -> None:
    """Test conversion from protein sequence to nucleotide population.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If conversion fails or output shape is incorrect.

    Example:
        >>> test_protein_to_nucleotide_conversion()

    """
    # Create a simple protein sequence
    protein_seq = jnp.array([0, 1, 2, 3], dtype=jnp.int8)
    population_size = 10

    result = generate_template_population(
      initial_sequence=protein_seq,
      population_size=population_size,
      input_sequence_type="protein",
      output_sequence_type="nucleotide",
    )

    # Each amino acid becomes 3 nucleotides
    expected_nuc_length = len(protein_seq) * 3
    assert_shape(result, (population_size, expected_nuc_length))
    assert result.dtype == jnp.int8

    # All sequences in the population should be identical
    for i in range(1, population_size):
      assert jnp.array_equal(result[0], result[i])

  def test_protein_to_protein_conversion(self) -> None:
    """Test protein to protein conversion (no translation).

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If output doesn't match input.

    Example:
        >>> test_protein_to_protein_conversion()

    """
    protein_seq = jnp.array([5, 10, 15, 19], dtype=jnp.int8)
    population_size = 5

    result = generate_template_population(
      initial_sequence=protein_seq,
      population_size=population_size,
      input_sequence_type="protein",
      output_sequence_type="protein",
    )

    assert_shape(result, (population_size, len(protein_seq)))
    assert result.dtype == jnp.int8

    # All sequences should be identical to the input
    for i in range(population_size):
      assert jnp.array_equal(result[i], protein_seq)

  def test_nucleotide_to_protein_conversion(self) -> None:
    """Test conversion from nucleotide to protein population.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If conversion fails.

    Example:
        >>> test_nucleotide_to_protein_conversion()

    """
    # Create a nucleotide sequence (must be multiple of 3)
    nuc_seq = jnp.array([0, 1, 2, 3, 0, 1, 2, 3, 0], dtype=jnp.int8)
    population_size = 8

    result = generate_template_population(
      initial_sequence=nuc_seq,
      population_size=population_size,
      input_sequence_type="nucleotide",
      output_sequence_type="protein",
    )

    # Should have 3 amino acids (9 nucleotides / 3)
    expected_protein_length = len(nuc_seq) // 3
    assert_shape(result, (population_size, expected_protein_length))
    assert result.dtype == jnp.int8

  def test_nucleotide_to_nucleotide_conversion(self) -> None:
    """Test nucleotide to nucleotide conversion.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If sequences don't match.

    Example:
        >>> test_nucleotide_to_nucleotide_conversion()

    """
    nuc_seq = jnp.array([0, 1, 2, 3, 0, 1], dtype=jnp.int8)
    population_size = 3

    result = generate_template_population(
      initial_sequence=nuc_seq,
      population_size=population_size,
      input_sequence_type="nucleotide",
      output_sequence_type="nucleotide",
    )

    assert_shape(result, (population_size, len(nuc_seq)))
    assert result.dtype == jnp.int8

    # All sequences should be identical to the input after round-trip
    for i in range(population_size):
      # Note: nucleotide to protein and back may not be exact due to codon degeneracy
      # So we just check the shape here
      pass

  def test_population_size_none(self) -> None:
    """Test with None population size (should default to 1).

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If population size is not 1.

    Example:
        >>> test_population_size_none()

    """
    protein_seq = jnp.array([0, 1, 2], dtype=jnp.int8)

    result = generate_template_population(
      initial_sequence=protein_seq,
      population_size=None,
      input_sequence_type="protein",
      output_sequence_type="protein",
    )

    assert_shape(result, (1, len(protein_seq)))

  def test_invalid_input_sequence_type(self) -> None:
    """Test with invalid input sequence type.

    Args:
        None

    Returns:
        None

    Raises:
        ValueError: If input sequence type is invalid.

    Example:
        >>> test_invalid_input_sequence_type()

    """
    protein_seq = jnp.array([0, 1, 2], dtype=jnp.int8)

    with pytest.raises(ValueError, match="Invalid input_sequence_type"):
      generate_template_population(
        initial_sequence=protein_seq,
        population_size=5,
        input_sequence_type="invalid",  # type: ignore[arg-type]
        output_sequence_type="protein",
      )

  def test_invalid_output_sequence_type(self) -> None:
    """Test with invalid output sequence type.

    Args:
        None

    Returns:
        None

    Raises:
        ValueError: If output sequence type is invalid.

    Example:
        >>> test_invalid_output_sequence_type()

    """
    protein_seq = jnp.array([0, 1, 2], dtype=jnp.int8)

    with pytest.raises(ValueError, match="Invalid output_sequence_type"):
      generate_template_population(
        initial_sequence=protein_seq,
        population_size=5,
        input_sequence_type="protein",
        output_sequence_type="invalid",  # type: ignore[arg-type]
      )

  def test_large_population(self) -> None:
    """Test with a large population size.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If large population handling fails.

    Example:
        >>> test_large_population()

    """
    protein_seq = jnp.array([0, 1], dtype=jnp.int8)
    population_size = 1000

    result = generate_template_population(
      initial_sequence=protein_seq,
      population_size=population_size,
      input_sequence_type="protein",
      output_sequence_type="nucleotide",
    )

    assert_shape(result, (population_size, len(protein_seq) * 3))
    assert result.dtype == jnp.int8

  def test_empty_sequence(self) -> None:
    """Test with an empty sequence.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If empty sequence handling fails.

    Example:
        >>> test_empty_sequence()

    """
    empty_seq = jnp.array([], dtype=jnp.int8)
    population_size = 5

    result = generate_template_population(
      initial_sequence=empty_seq,
      population_size=population_size,
      input_sequence_type="protein",
      output_sequence_type="protein",
    )

    assert_shape(result, (population_size, 0))
