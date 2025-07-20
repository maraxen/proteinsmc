"""Unit tests for the fitness models.

Tests FitnessFunction, CombineFunction, and FitnessEvaluator dataclasses,
including validation, filtering, and translation logic.

Run with:
  pytest tests/models/test_fitness.py
"""

import pytest

import dataclasses

import jax.numpy as jnp

from proteinsmc.models.fitness import (
  CombineFunction,
  FitnessEvaluator,
  FitnessFunction,
)


def test_fitness_function_dataclass():
  """Test initialization and immutability of FitnessFunction.

  Args:
    None
  Returns:
    None
  Raises:
    AssertionError: If the dataclass fields are not set correctly or are mutable.

  Example:
    >>> test_fitness_function_dataclass()

  """
  ff = FitnessFunction(name="mock", n_states=20, kwargs={"foo": 1})
  assert ff.name == "mock"
  assert ff.n_states == 20
  assert ff.kwargs == {"foo": 1}
  with pytest.raises(dataclasses.FrozenInstanceError):
    ff.name = "other"  # type: ignore[unreachable]


def test_combine_function_dataclass():
  """Test initialization and immutability of CombineFunction.

  Args:
    None
  Returns:
    None
  Raises:
    AssertionError: If the dataclass fields are not set correctly or are mutable.

  Example:
    >>> test_combine_function_dataclass()

  """
  cf = CombineFunction(name="sum", kwargs={"bar": 2})
  assert cf.name == "sum"
  assert cf.kwargs == {"bar": 2}
  with pytest.raises(dataclasses.FrozenInstanceError):
    cf.name = "other"  # type: ignore[unreachable]


def test_fitness_evaluator_requires_at_least_one_function():
  """Test that FitnessEvaluator raises ValueError if no fitness functions are provided.

  Args:
    None
  Returns:
    None
  Raises:
    AssertionError: If ValueError is not raised.

  Example:
    >>> test_fitness_evaluator_requires_at_least_one_function()

  """
  with pytest.raises(ValueError, match="At least one fitness function must be provided."):
    FitnessEvaluator(fitness_functions=())


def test_fitness_evaluator_get_functions_by_states():
  """Test filtering of fitness functions by n_states.

  Args:
    None
  Returns:
    None
  Raises:
    AssertionError: If filtering does not return the correct subset.

  Example:
    >>> test_fitness_evaluator_get_functions_by_states()

  """
  ff1 = FitnessFunction(name="f1", n_states=20)
  ff2 = FitnessFunction(name="f2", n_states=61)
  ff3 = FitnessFunction(name="f3", n_states=20)
  evaluator = FitnessEvaluator(fitness_functions=(ff1, ff2, ff3))
  result = evaluator.get_functions_by_states(20)
  assert result == [ff1, ff3]
  result2 = evaluator.get_functions_by_states(61)
  assert result2 == [ff2]


def test_fitness_evaluator_needs_translation():
  """Test needs_translation returns correct boolean array.

  Args:
    None
  Returns:
    None
  Raises:
    AssertionError: If the translation mask is incorrect.

  Example:
    >>> test_fitness_evaluator_needs_translation()

  """
  ff1 = FitnessFunction(name="f1", n_states=20)
  ff2 = FitnessFunction(name="f2", n_states=61)
  ff3 = FitnessFunction(name="f3", n_states=20)
  evaluator = FitnessEvaluator(fitness_functions=(ff1, ff2, ff3))
  mask = evaluator.needs_translation(20)
  expected = jnp.array([False, True, False])
  assert jnp.all(mask == expected), f"Expected {expected}, got {mask}"
  mask2 = evaluator.needs_translation(61)
  expected2 = jnp.array([True, False, True])
  assert jnp.all(mask2 == expected2), f"Expected {expected2}, got {mask2}"


def test_fitness_evaluator_default_combine_fn():
  """Test that the default combine_fn is set to 'sum'.

  Args:
    None
  Returns:
    None
  Raises:
    AssertionError: If the default combine_fn is not 'sum'.

  Example:
    >>> test_fitness_evaluator_default_combine_fn()

  """
  ff = FitnessFunction(name="f1", n_states=20)
  evaluator = FitnessEvaluator(fitness_functions=(ff,))
  assert evaluator.combine_fn.name == "sum"
