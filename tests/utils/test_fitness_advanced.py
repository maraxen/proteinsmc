"""Advanced tests for fitness function composition and chunking."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from chex import assert_shape

from proteinsmc.models.fitness import CombineFunction, FitnessEvaluator, FitnessFunction
from proteinsmc.models.types import EvoSequence
from proteinsmc.utils.fitness import get_fitness_function


def _identity_translate(sequence: EvoSequence, _key=None, _context=None) -> EvoSequence:  # type: ignore[misc]
  """Simple identity translation function for testing."""
  return sequence


class TestChunkedFitnessEvaluation:
  """Test chunked fitness evaluation."""

  def test_chunked_vs_non_chunked(self) -> None:
    """Test that chunked evaluation produces same results as non-chunked.
    
    Args:
        None
    
    Returns:
        None
    
    Raises:
        AssertionError: If chunked and non-chunked results differ.
    
    Example:
        >>> test_chunked_vs_non_chunked()
    
    """
    fitness_fn_config = FitnessFunction(name="cai", n_states=4)
    combine_fn_config = CombineFunction(name="sum")
    evaluator = FitnessEvaluator(
      fitness_functions=(fitness_fn_config,),
      combine_fn=combine_fn_config,
    )

    # Create fitness function without chunking
    fitness_fn_no_chunk = get_fitness_function(
      evaluator_config=evaluator,
      n_states=4,
      translate_func=_identity_translate,  # type: ignore[arg-type]
      batch_size=None,
    )

    # Create fitness function with chunking
    fitness_fn_chunked = get_fitness_function(
      evaluator_config=evaluator,
      n_states=4,
      translate_func=_identity_translate,  # type: ignore[arg-type]
      batch_size=2,
    )

    key = jax.random.PRNGKey(42)
    # Use length 6 sequences (multiple of 3 for nucleotide->protein translation)
    sequence = jnp.array(
      [[0, 1, 2, 3, 0, 1], [1, 2, 3, 0, 1, 2], [2, 3, 0, 1, 2, 3], [3, 0, 1, 2, 3, 0]],
      dtype=jnp.int8,
    )

    result_no_chunk = fitness_fn_no_chunk(key, sequence, None)
    result_chunked = fitness_fn_chunked(key, sequence, None)

    assert_shape(result_no_chunk, result_chunked.shape)
    assert jnp.allclose(result_no_chunk, result_chunked)

  def test_chunked_with_large_population(self) -> None:
    """Test chunked evaluation with large population.
    
    Args:
        None
    
    Returns:
        None
    
    Raises:
        AssertionError: If chunked evaluation fails for large population.
    
    Example:
        >>> test_chunked_with_large_population()
    
    """
    fitness_fn_config = FitnessFunction(name="cai", n_states=4)
    combine_fn_config = CombineFunction(name="sum")
    evaluator = FitnessEvaluator(
      fitness_functions=(fitness_fn_config,),
      combine_fn=combine_fn_config,
    )

    fitness_fn = get_fitness_function(
      evaluator_config=evaluator,
      n_states=4,
      translate_func=_identity_translate,  # type: ignore[arg-type]
      batch_size=5,
    )

    key = jax.random.PRNGKey(42)
    # Create a large population with length 6 sequences (multiple of 3)
    sequence = jnp.array(
      [[i % 4, (i + 1) % 4, (i + 2) % 4, (i + 3) % 4, i % 4, (i + 1) % 4] for i in range(20)],
      dtype=jnp.int8,
    )

    result = fitness_fn(key, sequence, None)

    # Should have one row per sequence + components
    assert_shape(result, (2, 20))

  def test_chunked_with_different_batch_sizes(self) -> None:
    """Test chunked evaluation with different chunk sizes.
    
    Args:
        None
    
    Returns:
        None
    
    Raises:
        AssertionError: If different chunk sizes produce different results.
    
    Example:
        >>> test_chunked_with_different_batch_sizes()
    
    """
    fitness_fn_config = FitnessFunction(name="cai", n_states=4)
    combine_fn_config = CombineFunction(name="sum")
    evaluator = FitnessEvaluator(
      fitness_functions=(fitness_fn_config,),
      combine_fn=combine_fn_config,
    )

    key = jax.random.PRNGKey(42)
    # Use length 6 sequences (multiple of 3)
    sequence = jnp.array(
      [[0, 1, 2, 3, 0, 1], [1, 2, 3, 0, 1, 2], [2, 3, 0, 1, 2, 3], [3, 0, 1, 2, 3, 0]],
      dtype=jnp.int8,
    )

    results = []
    for batch_size in [1, 2, 4]:
      fitness_fn = get_fitness_function(
        evaluator_config=evaluator,
        n_states=4,
        translate_func=_identity_translate,  # type: ignore[arg-type]
        batch_size=batch_size,
      )
      result = fitness_fn(key, sequence, None)
      results.append(result)

    # All results should be the same
    for i in range(1, len(results)):
      assert jnp.allclose(results[0], results[i])


class TestMultipleFitnessComponents:
  """Test evaluation with multiple fitness components."""

  def test_multiple_fitness_functions(self) -> None:
    """Test combining multiple fitness functions.
    
    Args:
        None
    
    Returns:
        None
    
    Raises:
        AssertionError: If multiple fitness functions fail to combine.
    
    Example:
        >>> test_multiple_fitness_functions()
    
    """
    fitness_fn_config1 = FitnessFunction(name="cai", n_states=4)
    fitness_fn_config2 = FitnessFunction(name="cai", n_states=4)
    combine_fn_config = CombineFunction(name="sum")
    evaluator = FitnessEvaluator(
      fitness_functions=(fitness_fn_config1, fitness_fn_config2),
      combine_fn=combine_fn_config,
    )

    fitness_fn = get_fitness_function(
      evaluator_config=evaluator,
      n_states=4,
      translate_func=_identity_translate,  # type: ignore[arg-type]
    )

    key = jax.random.PRNGKey(42)
    # Use length 6 sequences (multiple of 3)
    sequence = jnp.array([[0, 1, 2, 3, 0, 1], [1, 2, 3, 0, 1, 2]], dtype=jnp.int8)

    result = fitness_fn(key, sequence, None)

    # Should have combined fitness + 2 components
    assert_shape(result, (3, 2))

  def test_weighted_combine_function(self) -> None:
    """Test weighted combine function.
    
    Args:
        None
    
    Returns:
        None
    
    Raises:
        AssertionError: If weighted combine function fails.
    
    Example:
        >>> test_weighted_combine_function()
    
    """
    fitness_fn_config1 = FitnessFunction(name="cai", n_states=4)
    fitness_fn_config2 = FitnessFunction(name="cai", n_states=4)
    combine_fn_config = CombineFunction(name="weighted_sum", kwargs={"weights": [0.7, 0.3]})
    evaluator = FitnessEvaluator(
      fitness_functions=(fitness_fn_config1, fitness_fn_config2),
      combine_fn=combine_fn_config,
    )

    fitness_fn = get_fitness_function(
      evaluator_config=evaluator,
      n_states=4,
      translate_func=_identity_translate,  # type: ignore[arg-type]
    )

    key = jax.random.PRNGKey(42)
    # Use length 6 sequences (multiple of 3)
    sequence = jnp.array([[0, 1, 2, 3, 0, 1], [1, 2, 3, 0, 1, 2]], dtype=jnp.int8)

    result = fitness_fn(key, sequence, None)

    # Should have combined fitness + 2 components
    assert_shape(result, (3, 2))


class TestContextDependentFitness:
  """Test fitness evaluation with context (beta) parameter."""

  def test_fitness_with_context(self) -> None:
    """Test fitness evaluation scales with context parameter.
    
    Args:
        None
    
    Returns:
        None
    
    Raises:
        AssertionError: If context parameter doesn't scale fitness.
    
    Example:
        >>> test_fitness_with_context()
    
    """
    fitness_fn_config = FitnessFunction(name="cai", n_states=4)
    combine_fn_config = CombineFunction(name="sum")
    evaluator = FitnessEvaluator(
      fitness_functions=(fitness_fn_config,),
      combine_fn=combine_fn_config,
    )

    fitness_fn = get_fitness_function(
      evaluator_config=evaluator,
      n_states=4,
      translate_func=_identity_translate,  # type: ignore[arg-type]
    )

    key = jax.random.PRNGKey(42)
    # Use length 6 sequences (multiple of 3)
    sequence = jnp.array([[0, 1, 2, 3, 0, 1], [1, 2, 3, 0, 1, 2]], dtype=jnp.int8)

    result_no_context = fitness_fn(key, sequence, None)
    result_with_context = fitness_fn(key, sequence, jnp.array(0.5))

    # With context, combined fitness should be scaled
    # Components remain unchanged
    assert_shape(result_no_context, result_with_context.shape)
    # Combined fitness (first row) should be scaled
    assert jnp.allclose(result_no_context[0] * 0.5, result_with_context[0])
    # Components (other rows) should be unchanged
    assert jnp.allclose(result_no_context[1:], result_with_context[1:])

  def test_fitness_with_different_context_values(self) -> None:
    """Test fitness with different context values.
    
    Args:
        None
    
    Returns:
        None
    
    Raises:
        AssertionError: If different context values don't scale appropriately.
    
    Example:
        >>> test_fitness_with_different_context_values()
    
    """
    fitness_fn_config = FitnessFunction(name="cai", n_states=4)
    combine_fn_config = CombineFunction(name="sum")
    evaluator = FitnessEvaluator(
      fitness_functions=(fitness_fn_config,),
      combine_fn=combine_fn_config,
    )

    fitness_fn = get_fitness_function(
      evaluator_config=evaluator,
      n_states=4,
      translate_func=_identity_translate,  # type: ignore[arg-type]
    )

    key = jax.random.PRNGKey(42)
    # Use length 6 sequences (multiple of 3)
    sequence = jnp.array([[0, 1, 2, 3, 0, 1], [1, 2, 3, 0, 1, 2]], dtype=jnp.int8)

    result_base = fitness_fn(key, sequence, jnp.array(1.0))

    for beta in [0.1, 0.5, 2.0]:
      result = fitness_fn(key, sequence, jnp.array(beta))
      # Combined fitness should scale linearly with beta
      assert jnp.allclose(result_base[0] * beta, result[0])


class TestTranslationIntegration:
  """Test fitness functions that require translation."""

  def test_needs_translation(self) -> None:
    """Test that needs_translation is correctly determined.
    
    Args:
        None
    
    Returns:
        None
    
    Raises:
        AssertionError: If needs_translation is incorrect.
    
    Example:
        >>> test_needs_translation()
    
    """
    # Nucleotide fitness functions need translation
    fitness_fn_config = FitnessFunction(name="cai", n_states=4)
    combine_fn_config = CombineFunction(name="sum")
    evaluator = FitnessEvaluator(
      fitness_functions=(fitness_fn_config,),
      combine_fn=combine_fn_config,
    )

    # Check needs_translation method
    needs_trans = evaluator.needs_translation(n_states=4)
    # Should be a JAX array, not a list
    assert isinstance(needs_trans, jnp.ndarray)
    assert needs_trans.shape[0] == 1

  def test_fitness_with_translation(self) -> None:
    """Test fitness evaluation with custom translation function.
    
    Args:
        None
    
    Returns:
        None
    
    Raises:
        AssertionError: If translation doesn't work correctly.
    
    Example:
        >>> test_fitness_with_translation()
    
    """

    def custom_translate(sequence: EvoSequence, _key=None, _context=None) -> EvoSequence:  # type: ignore[misc]
      """Custom translation that doubles values."""
      return sequence * 2

    fitness_fn_config = FitnessFunction(name="cai", n_states=4)
    combine_fn_config = CombineFunction(name="sum")
    evaluator = FitnessEvaluator(
      fitness_functions=(fitness_fn_config,),
      combine_fn=combine_fn_config,
    )

    fitness_fn = get_fitness_function(
      evaluator_config=evaluator,
      n_states=4,
      translate_func=custom_translate,  # type: ignore[arg-type]
    )

    key = jax.random.PRNGKey(42)
    # Use length 6 sequences (multiple of 3)
    sequence = jnp.array([[0, 1, 2, 3, 0, 1], [1, 2, 3, 0, 1, 2]], dtype=jnp.int8)

    result = fitness_fn(key, sequence, None)

    # Should successfully evaluate
    assert_shape(result, (2, 2))
