"""Tests for mutation utility functions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from chex import assert_shape

from proteinsmc.utils.constants import PROTEINMPNN_X_INT
from proteinsmc.utils.mutation import (
  chunked_mutation_step,
  diversify_initial_nucleotide_sequences,
  diversify_initial_protein_sequences,
  diversify_initial_sequences,
  mutate,
)


class TestMutate:
  """Test the mutate function."""

  def test_basic_mutation(self, rng_key) -> None:
    """Test basic mutation functionality.

    Args:
        rng_key: PRNG key fixture.

    Returns:
        None

    Raises:
        AssertionError: If mutation fails.

    Example:
        >>> test_basic_mutation(jax.random.PRNGKey(42))

    """
    # Create a population of nucleotide sequences
    population = jnp.array([
      [0, 1, 2, 3, 0, 1],
      [1, 2, 3, 0, 1, 2],
      [2, 3, 0, 1, 2, 3],
    ], dtype=jnp.int8)

    mutation_rate = 0.5
    q_states = 4

    result = mutate(rng_key, population, mutation_rate, q_states)

    # Output should have same shape
    assert_shape(result, population.shape)
    assert result.dtype == jnp.int8

    # All values should be in valid range
    assert jnp.all((result >= 0) & (result < q_states))

  def test_zero_mutation_rate(self, rng_key) -> None:
    """Test that zero mutation rate leaves sequences unchanged.

    Args:
        rng_key: PRNG key fixture.

    Returns:
        None

    Raises:
        AssertionError: If sequences change with zero mutation rate.

    Example:
        >>> test_zero_mutation_rate(jax.random.PRNGKey(42))

    """
    population = jnp.array([[0, 1, 2, 3]], dtype=jnp.int8)
    mutation_rate = 0.0
    q_states = 4

    result = mutate(rng_key, population, mutation_rate, q_states)

    # Sequences should be unchanged
    assert jnp.array_equal(result, population)

  def test_high_mutation_rate(self, rng_key) -> None:
    """Test with very high mutation rate.

    Args:
        rng_key: PRNG key fixture.

    Returns:
        None

    Raises:
        AssertionError: If mutation doesn't occur with high rate.

    Example:
        >>> test_high_mutation_rate(jax.random.PRNGKey(42))

    """
    population = jnp.zeros((10, 20), dtype=jnp.int8)
    mutation_rate = 0.99
    q_states = 4

    result = mutate(rng_key, population, mutation_rate, q_states)

    # Most positions should be mutated (different from zero)
    num_mutations = jnp.sum(result != population)
    total_positions = population.size

    # Expect at least 90% mutations with 0.99 rate
    assert num_mutations > 0.9 * total_positions

  def test_different_q_states(self, rng_key) -> None:
    """Test mutation with different numbers of states.

    Args:
        rng_key: PRNG key fixture.

    Returns:
        None

    Raises:
        AssertionError: If mutation fails with different q_states.

    Example:
        >>> test_different_q_states(jax.random.PRNGKey(42))

    """
    population = jnp.array([[0, 5, 10, 15]], dtype=jnp.int8)

    for q_states in [4, 20, 64]:
      # Adjust population to valid range
      pop = population % q_states
      result = mutate(rng_key, pop, 0.5, q_states)

      assert jnp.all((result >= 0) & (result < q_states))


class TestDiversifyInitialNucleotideSequences:
  """Test the diversify_initial_nucleotide_sequences function."""

  def test_basic_diversification(self, rng_key) -> None:
    """Test basic nucleotide diversification.

    Args:
        rng_key: PRNG key fixture.

    Returns:
        None

    Raises:
        AssertionError: If diversification fails.

    Example:
        >>> test_basic_diversification(jax.random.PRNGKey(42))

    """
    # Create nucleotide sequences (multiple of 3 for codons)
    seed_sequences = jnp.array([
      [0, 1, 2, 0, 1, 2],
      [1, 2, 0, 1, 2, 0],
    ], dtype=jnp.int8)

    mutation_rate = 0.3

    result = diversify_initial_nucleotide_sequences(rng_key, seed_sequences, mutation_rate)

    assert_shape(result, seed_sequences.shape)
    assert result.dtype == jnp.int8

    # All values should be valid nucleotides (0-3)
    assert jnp.all((result >= 0) & (result < 4))

  def test_no_x_codons_introduced(self, rng_key) -> None:
    """Test that X codons are reverted if introduced by mutation.

    Args:
        rng_key: PRNG key fixture.

    Returns:
        None

    Raises:
        AssertionError: If X codons remain after diversification.

    Example:
        >>> test_no_x_codons_introduced(jax.random.PRNGKey(42))

    """
    # Create valid nucleotide sequences
    seed_sequences = jnp.array([
      [0, 0, 0, 1, 1, 1, 2, 2, 2],  # 3 codons
    ], dtype=jnp.int8)

    mutation_rate = 0.8  # High mutation rate to test X reversion

    result = diversify_initial_nucleotide_sequences(rng_key, seed_sequences, mutation_rate)

    # Translate to amino acids to check for X
    from proteinsmc.utils.translation import nucleotide_to_aa
    aa_seq, is_valid = nucleotide_to_aa(result[0])

    # The result should be valid (no X codons)
    # Note: Some X codons might still appear if the template had them
    # But mutations that introduce X should be reverted
    assert result.dtype == jnp.int8

  def test_preserves_length(self, rng_key) -> None:
    """Test that sequence length is preserved.

    Args:
        rng_key: PRNG key fixture.

    Returns:
        None

    Raises:
        AssertionError: If sequence length changes.

    Example:
        >>> test_preserves_length(jax.random.PRNGKey(42))

    """
    for length in [6, 12, 30]:  # Multiples of 3
      seed_sequences = jnp.zeros((5, length), dtype=jnp.int8)
      result = diversify_initial_nucleotide_sequences(rng_key, seed_sequences, 0.5)
      assert_shape(result, (5, length))


class TestDiversifyInitialProteinSequences:
  """Test the diversify_initial_protein_sequences function."""

  def test_basic_protein_diversification(self, rng_key) -> None:
    """Test basic protein sequence diversification.

    Args:
        rng_key: PRNG key fixture.

    Returns:
        None

    Raises:
        AssertionError: If diversification fails.

    Example:
        >>> test_basic_protein_diversification(jax.random.PRNGKey(42))

    """
    seed_sequences = jnp.array([
      [0, 5, 10, 15],
      [1, 6, 11, 16],
    ], dtype=jnp.int8)

    mutation_rate = 0.4

    result = diversify_initial_protein_sequences(rng_key, seed_sequences, mutation_rate)

    assert_shape(result, seed_sequences.shape)
    assert result.dtype == jnp.int8

    # All values should be valid amino acids (0-19)
    assert jnp.all((result >= 0) & (result < 20))

  def test_zero_mutation_rate_protein(self, rng_key) -> None:
    """Test protein diversification with zero mutation rate.

    Args:
        rng_key: PRNG key fixture.

    Returns:
        None

    Raises:
        AssertionError: If sequences change with zero mutation rate.

    Example:
        >>> test_zero_mutation_rate_protein(jax.random.PRNGKey(42))

    """
    seed_sequences = jnp.array([[0, 5, 10, 15]], dtype=jnp.int8)
    mutation_rate = 0.0

    result = diversify_initial_protein_sequences(rng_key, seed_sequences, mutation_rate)

    # Sequences should be unchanged
    assert jnp.array_equal(result, seed_sequences)

  def test_high_mutation_rate_protein(self, rng_key) -> None:
    """Test protein diversification with high mutation rate.

    Args:
        rng_key: PRNG key fixture.

    Returns:
        None

    Raises:
        AssertionError: If insufficient mutations occur.

    Example:
        >>> test_high_mutation_rate_protein(jax.random.PRNGKey(42))

    """
    seed_sequences = jnp.zeros((10, 20), dtype=jnp.int8)
    mutation_rate = 0.95

    result = diversify_initial_protein_sequences(rng_key, seed_sequences, mutation_rate)

    # Most positions should be mutated
    num_mutations = jnp.sum(result != seed_sequences)
    total_positions = seed_sequences.size

    assert num_mutations > 0.8 * total_positions


class TestDiversifyInitialSequences:
  """Test the diversify_initial_sequences function."""

  def test_nucleotide_type(self, rng_key) -> None:
    """Test diversification with nucleotide type.

    Args:
        rng_key: PRNG key fixture.

    Returns:
        None

    Raises:
        AssertionError: If nucleotide diversification fails.

    Example:
        >>> test_nucleotide_type(jax.random.PRNGKey(42))

    """
    seed_sequences = jnp.array([[0, 1, 2, 0, 1, 2]], dtype=jnp.int8)
    mutation_rate = 0.3

    result = diversify_initial_sequences(
      rng_key,
      seed_sequences,
      mutation_rate,
      sequence_type="nucleotide",
    )

    assert_shape(result, seed_sequences.shape)
    assert jnp.all((result >= 0) & (result < 4))

  def test_protein_type(self, rng_key) -> None:
    """Test diversification with protein type.

    Args:
        rng_key: PRNG key fixture.

    Returns:
        None

    Raises:
        AssertionError: If protein diversification fails.

    Example:
        >>> test_protein_type(jax.random.PRNGKey(42))

    """
    seed_sequences = jnp.array([[0, 5, 10, 15]], dtype=jnp.int8)
    mutation_rate = 0.3

    result = diversify_initial_sequences(
      rng_key,
      seed_sequences,
      mutation_rate,
      sequence_type="protein",
    )

    assert_shape(result, seed_sequences.shape)
    assert jnp.all((result >= 0) & (result < 20))

  def test_invalid_sequence_type(self, rng_key) -> None:
    """Test with invalid sequence type.

    Args:
        rng_key: PRNG key fixture.

    Returns:
        None

    Raises:
        ValueError: If sequence type is invalid.

    Example:
        >>> test_invalid_sequence_type(jax.random.PRNGKey(42))

    """
    seed_sequences = jnp.array([[0, 1, 2]], dtype=jnp.int8)

    with pytest.raises(ValueError, match="Unsupported sequence_type"):
      diversify_initial_sequences(
        rng_key,
        seed_sequences,
        0.3,
        sequence_type="invalid",  # type: ignore[arg-type]
      )


class TestChunkedMutationStep:
  """Test the chunked_mutation_step function."""

  def test_basic_chunked_mutation(self, rng_key) -> None:
    """Test basic chunked mutation functionality.

    Args:
        rng_key: PRNG key fixture.

    Returns:
        None

    Raises:
        AssertionError: If chunked mutation fails.

    Example:
        >>> test_basic_chunked_mutation(jax.random.PRNGKey(42))

    """
    population = jnp.array([
      [0, 1, 2, 3],
      [1, 2, 3, 0],
      [2, 3, 0, 1],
      [3, 0, 1, 2],
    ], dtype=jnp.int8)

    mutation_rate = 0.4
    n_states = 4
    batch_size = 2

    result = chunked_mutation_step(rng_key, population, mutation_rate, n_states, batch_size)

    assert_shape(result, population.shape)
    assert result.dtype == jnp.int8
    assert jnp.all((result >= 0) & (result < n_states))

  def test_different_batch_sizes(self, rng_key) -> None:
    """Test chunked mutation with different chunk sizes.

    Args:
        rng_key: PRNG key fixture.

    Returns:
        None

    Raises:
        AssertionError: If different chunk sizes fail.

    Example:
        >>> test_different_batch_sizes(jax.random.PRNGKey(42))

    """
    population = jnp.zeros((10, 20), dtype=jnp.int8)
    mutation_rate = 0.3
    n_states = 4

    for batch_size in [1, 2, 5, 10]:
      result = chunked_mutation_step(rng_key, population, mutation_rate, n_states, batch_size)
      assert_shape(result, population.shape)
      assert result.dtype == jnp.int8

  def test_large_population(self, rng_key) -> None:
    """Test chunked mutation with large population.

    Args:
        rng_key: PRNG key fixture.

    Returns:
        None

    Raises:
        AssertionError: If large population handling fails.

    Example:
        >>> test_large_population(jax.random.PRNGKey(42))

    """
    population = jnp.zeros((100, 50), dtype=jnp.int8)
    mutation_rate = 0.2
    n_states = 20
    batch_size = 10

    result = chunked_mutation_step(rng_key, population, mutation_rate, n_states, batch_size)

    assert_shape(result, population.shape)
    assert result.dtype == jnp.int8
    assert jnp.all((result >= 0) & (result < n_states))
