"""Unit tests for FitnessFunction and FitnessEvaluator data models.

Tests cover initialization, type checking, and edge cases for FitnessFunction and FitnessEvaluator.
"""
import pytest
from proteinsmc.models import FitnessFunction, FitnessEvaluator


def test_fitness_function_initialization():
  """Test FitnessFunction initialization with valid arguments.
  Args:
    None
  Returns:
    None
  Raises:
    AssertionError: If the fields do not match expected values.
  Example:
    >>> test_fitness_function_initialization()
  """
  func = FitnessFunction(name="stability", n_states=20)
  assert func.name == "stability"
  assert func.n_states == 20


def test_fitness_evaluator_initialization():
  """Test FitnessEvaluator initialization with valid fitness functions.
  Args:
    None
  Returns:
    None
  Raises:
    AssertionError: If the evaluator does not contain the expected functions.
  Example:
    >>> test_fitness_evaluator_initialization()
  """
  evaluator = FitnessEvaluator(fitness_functions=(FitnessFunction(name="stability", n_states=20),))
  assert isinstance(evaluator.fitness_functions, tuple)
  assert evaluator.fitness_functions[0].name == "stability"


def test_fitness_function_invalid_type():
  """Test FitnessFunction raises error on invalid types.
  Args:
    None
  Returns:
    None
  Raises:
    AssertionError: If TypeError is not raised for invalid types.
  Example:
    >>> test_fitness_function_invalid_type()
  """
  with pytest.raises(TypeError):
    FitnessFunction(name=123, n_states=20)  # type: ignore[arg-type]
    
  with pytest.raises(TypeError):
    FitnessFunction(name="test_function", n_states="twenty")  # type: ignore[arg-type]
    
