"""Tests for fitness utility functions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from chex import assert_shape

from proteinsmc.models.fitness import CombineFunction, FitnessEvaluator, FitnessFunction
from proteinsmc.models.types import EvoSequence
from proteinsmc.utils.fitness import (
  COMBINE_FUNCTIONS,
  FITNESS_FUNCTIONS,
  get_fitness_function,
)


def _identity_translate(*, sequence: EvoSequence, **_kwargs) -> EvoSequence:  # type: ignore[misc]
  """Simple identity translation function for testing."""
  return sequence


class TestGetFitnessFunction:
  """Test the get_fitness_function factory."""

  def test_function_creation(self) -> None:
    """Test creating a fitness function.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If fitness function creation fails.

    Example:
        >>> test_function_creation()

    """
    # Test that the function can be created
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

    # Verify the function exists and is callable
    assert callable(fitness_fn)

  def test_unknown_fitness_function(self) -> None:
    """Test with unknown fitness function name.

    Args:
        None

    Returns:
        None

    Raises:
        ValueError: If fitness function is unknown.

    Example:
        >>> test_unknown_fitness_function()

    """
    fitness_fn_config = FitnessFunction(name="unknown_function", n_states=4)
    combine_fn_config = CombineFunction(name="sum")
    evaluator = FitnessEvaluator(
      fitness_functions=(fitness_fn_config,),
      combine_fn=combine_fn_config,
    )

    def identity_translate(sequence, _key, _context):
      return sequence

    with pytest.raises(ValueError, match="Unknown fitness function"):
      get_fitness_function(
        evaluator_config=evaluator,
        n_states=4,
        translate_func=_identity_translate,  # type: ignore[arg-type]
      )

  def test_unknown_combine_function(self) -> None:
    """Test with unknown combine function name.

    Args:
        None

    Returns:
        None

    Raises:
        ValueError: If combine function is unknown.

    Example:
        >>> test_unknown_combine_function()

    """
    fitness_fn_config = FitnessFunction(name="cai", n_states=4)
    combine_fn_config = CombineFunction(name="unknown_combine")
    evaluator = FitnessEvaluator(
      fitness_functions=(fitness_fn_config,),
      combine_fn=combine_fn_config,
    )

    with pytest.raises(ValueError, match="Unknown combine function"):
      get_fitness_function(
        evaluator_config=evaluator,
        n_states=4,
        translate_func=_identity_translate,  # type: ignore[arg-type]
      )


class TestFitnessAndCombineDictionaries:
  """Test the FITNESS_FUNCTIONS and COMBINE_FUNCTIONS dictionaries."""

  def test_fitness_functions_dict_exists(self) -> None:
    """Test that FITNESS_FUNCTIONS dictionary is properly populated.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If dictionary is empty or invalid.

    Example:
        >>> test_fitness_functions_dict_exists()

    """
    assert isinstance(FITNESS_FUNCTIONS, dict)
    assert len(FITNESS_FUNCTIONS) > 0

    # Check expected functions are present
    assert "cai" in FITNESS_FUNCTIONS
    assert "mpnn" in FITNESS_FUNCTIONS
    assert "esm" in FITNESS_FUNCTIONS

  def test_combine_functions_dict_exists(self) -> None:
    """Test that COMBINE_FUNCTIONS dictionary is properly populated.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If dictionary is empty or invalid.

    Example:
        >>> test_combine_functions_dict_exists()

    """
    assert isinstance(COMBINE_FUNCTIONS, dict)
    assert len(COMBINE_FUNCTIONS) > 0

    # Check expected functions are present
    assert "sum" in COMBINE_FUNCTIONS
    assert "weighted_sum" in COMBINE_FUNCTIONS

  def test_fitness_functions_are_callable(self) -> None:
    """Test that all fitness functions in the dict are callable.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If any function is not callable.

    Example:
        >>> test_fitness_functions_are_callable()

    """
    for name, func in FITNESS_FUNCTIONS.items():
      assert callable(func), f"Fitness function '{name}' is not callable"

  def test_combine_functions_are_callable(self) -> None:
    """Test that all combine functions in the dict are callable.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If any function is not callable.

    Example:
        >>> test_combine_functions_are_callable()

    """
    for name, func in COMBINE_FUNCTIONS.items():
      assert callable(func), f"Combine function '{name}' is not callable"
