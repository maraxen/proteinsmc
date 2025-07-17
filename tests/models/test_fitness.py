"""Tests for fitness model classes."""

from __future__ import annotations

from typing import Any, Callable
from unittest.mock import Mock

import chex
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float, PRNGKeyArray
from proteinsmc.models.types import EvoSequence

from proteinsmc.models.fitness import (
  CombineFunction,
  CombineRegistry,
  CombineRegistryItem,
  FitnessFuncSignature,
  CombineFuncSignature,
  FitnessEvaluator,
  FitnessFunction,
  FitnessRegistry,
  FitnessRegistryItem,
)


# Mock fitness functions for testing


def make_mock_combine_function() -> CombineFuncSignature:
  """Create a mock combine function."""
  @jax.jit
  def mock_combine_func(fitness_scores: Array, _context: Array | None = None) -> Float:
    """Mock combine function that sums fitness scores."""
    return jnp.sum(fitness_scores)
  return mock_combine_func

def make_mock_fitness_function() -> FitnessFuncSignature:
  """Create a mock fitness function."""
  @jax.jit
  def mock_fitness_func(_key: PRNGKeyArray, sequence: EvoSequence, _context: Array | None = None) -> Float:
    """Mock fitness function that returns sequence length as fitness."""
    return jnp.array(len(sequence), dtype=jnp.float32)
  return mock_fitness_func

class TestFitnessRegistryItem:
  """Test cases for FitnessRegistryItem."""

  def test_init_success(self) -> None:
    """Test successful FitnessRegistryItem initialization."""
    item = FitnessRegistryItem(
      method_factory=make_mock_fitness_function,
      name="test_fitness",
      input_type="protein",
    )
    assert item.name == "test_fitness"
    assert item.input_type == "protein"
    assert callable(item.method_factory)

  def test_init_invalid_input_type(self) -> None:
    """Test FitnessRegistryItem raises ValueError for invalid input_type."""
    with pytest.raises(ValueError, match="Invalid input_type 'invalid'"):
      FitnessRegistryItem(
        method_factory=make_mock_fitness_function,
        name="test_fitness",
        input_type="invalid",  # type: ignore
      )

  def test_init_invalid_name_type(self) -> None:
    """Test FitnessRegistryItem raises TypeError for invalid name type."""
    with pytest.raises(TypeError, match="name must be a string"):
      FitnessRegistryItem(
        method_factory=make_mock_fitness_function,
        name=123,  # type: ignore
        input_type="protein",
      )

  def test_init_invalid_method_factory(self) -> None:
    """Test FitnessRegistryItem raises TypeError for non-callable factory."""
    with pytest.raises(TypeError, match="Fitness method .* is not callable"):
      FitnessRegistryItem(
        method_factory="not_callable",  # type: ignore
        name="test_fitness",
        input_type="protein",
      )

  def test_call_returns_callable(self) -> None:
    """Test that calling FitnessRegistryItem returns a callable."""
    item = FitnessRegistryItem(
      method_factory=make_mock_fitness_function,
      name="test_fitness",
      input_type="protein",
    )
    result = item()
    assert callable(result)


class TestFitnessRegistry:
  """Test cases for FitnessRegistry."""

  @pytest.fixture
  def sample_fitness_item(self) -> FitnessRegistryItem:
    """Create a sample fitness registry item."""
    return FitnessRegistryItem(
      method_factory=make_mock_fitness_function,
      name="sample_fitness",
      input_type="protein",
    )

  @pytest.fixture
  def sample_fitness_registry(
    self,
    sample_fitness_item: FitnessRegistryItem,
  ) -> FitnessRegistry:
    """Create a sample fitness registry."""
    return FitnessRegistry(items={"sample_fitness": sample_fitness_item})

  def test_init_success(self, sample_fitness_registry: FitnessRegistry) -> None:
    """Test successful FitnessRegistry initialization."""
    assert "sample_fitness" in sample_fitness_registry
    assert len(sample_fitness_registry.items) == 1

  def test_init_invalid_item_type(self) -> None:
    """Test FitnessRegistry raises TypeError for invalid item types."""
    with pytest.raises(
      TypeError,
      match="All items in FitnessRegistry must be instances of FitnessRegistryItem",
    ):
      FitnessRegistry(items={"invalid": "not_a_fitness_item"})  # type: ignore


class TestFitnessFunction:
  """Test cases for FitnessFunction."""

  def test_init_success(self) -> None:
    """Test successful FitnessFunction initialization."""
    ff = FitnessFunction(func="test_fitness", input_type="protein")
    assert ff.func == "test_fitness"
    assert ff.input_type == "protein"

  def test_init_invalid_input_type(self) -> None:
    """Test FitnessFunction raises ValueError for invalid input_type."""
    with pytest.raises(ValueError, match="Invalid input_type 'invalid'"):
      FitnessFunction(func="test_fitness", input_type="invalid")  # type: ignore

  def test_call_success(self) -> None:
    """Test successful call of FitnessFunction."""
    mock_registry = Mock(spec=FitnessRegistry)
    mock_item = Mock()
    mock_item.return_value = make_mock_fitness_function()
    mock_registry.__contains__.return_value = True
    mock_registry.get.return_value = mock_item

    ff = FitnessFunction(func="test_fitness")
    result_func = ff(mock_registry)

    mock_registry.get.assert_called_once_with("test_fitness")
    assert callable(result_func)

  def test_call_missing_function(self) -> None:
    """Test FitnessFunction call raises ValueError for missing function."""
    mock_registry = Mock(spec=FitnessRegistry)
    mock_registry.__contains__.return_value = False

    ff = FitnessFunction(func="missing_fitness")
    with pytest.raises(ValueError, match="Fitness function missing_fitness is not registered"):
      ff(mock_registry)

  def test_pytree_registration(self) -> None:
    """Test that FitnessFunction is properly registered as a PyTree."""
    ff = FitnessFunction(func="test_fitness", input_type="nucleotide")

    # Test that it can be used in JAX transformations
    def process_ff(ff_input: FitnessFunction) -> str:
      return ff_input.func

    jitted_process = jax.jit(process_ff)
    result = jitted_process(ff)
    assert result == "test_fitness"

    # Test tree flatten/unflatten
    leaves, treedef = jax.tree_util.tree_flatten(ff)
    unflattened_ff = jax.tree_util.tree_unflatten(treedef, leaves)

    assert ff.func == unflattened_ff.func
    assert ff.input_type == unflattened_ff.input_type


class TestCombineFunction:
  """Test cases for CombineFunction."""

  def test_init_success(self) -> None:
    """Test successful CombineFunction initialization."""
    cf = CombineFunction(func="sum_combine")
    assert cf.func == "sum_combine"


class TestCombineRegistryItem:
  """Test cases for CombineRegistryItem."""

  def test_init_success(self) -> None:
    """Test successful CombineRegistryItem initialization."""
    combine_func = CombineFunction(func="sum_combine")
    item = CombineRegistryItem(
      method_factory=make_mock_combine_function,
      name="sum_combine",
    )
    assert item.name == "sum_combine"
    assert item.func == combine_func
    assert callable(item.method_factory)

  def test_call_returns_callable(self) -> None:
    """Test that calling CombineRegistryItem returns a callable."""
    combine_func = CombineFunction(func="sum_combine")
    item = CombineRegistryItem(
      method_factory=make_mock_combine_function,
      name="sum_combine",
    )
    result = item()
    assert callable(result)


class TestCombineRegistry:
  """Test cases for CombineRegistry."""

  @pytest.fixture
  def sample_combine_item(self) -> CombineRegistryItem:
    """Create a sample combine registry item."""
    combine_func = CombineFunction(func="sum_combine")
    return CombineRegistryItem(
      method_factory=make_mock_combine_function,
      name="sum_combine",
    )

  @pytest.fixture
  def sample_combine_registry(
    self,
    sample_combine_item: CombineRegistryItem,
  ) -> CombineRegistry:
    """Create a sample combine registry."""
    registry = CombineRegistry(items={"sum_combine": sample_combine_item})
    return registry

  def test_get_success(
    self,
    sample_combine_registry: CombineRegistry,
    sample_combine_item: CombineRegistryItem,
  ) -> None:
    """Test successful get method."""
    retrieved_item = sample_combine_registry.get("sum_combine")
    assert retrieved_item == sample_combine_item

  def test_get_missing(self, sample_combine_registry: CombineRegistry) -> None:
    """Test get method raises KeyError for missing item."""
    with pytest.raises(KeyError, match="Combine function missing is not registered"):
      sample_combine_registry.get("missing")


class TestFitnessEvaluator:
  """Test cases for FitnessEvaluator."""

  @pytest.fixture
  def sample_fitness_functions(self) -> tuple[FitnessFunction, ...]:
    """Create sample fitness functions."""
    return (
      FitnessFunction(func="fitness1", input_type="protein"),
      FitnessFunction(func="fitness2", input_type="nucleotide"),
      FitnessFunction(func="fitness3", input_type="protein"),
    )
    
  @pytest.fixture
  def sample_fitness_args(
    self  ) -> dict[str, dict[str, Any]]:
    """Create sample fitness arguments."""
    return {
      "fitness1": {},
      "fitness2": {},
      "fitness3": {},
    }

  @pytest.fixture
  def sample_fitness_evaluator(
    self,
    sample_fitness_functions: tuple[FitnessFunction, ...],
    sample_fitness_args: dict[str, dict[str, Any]],
  ) -> FitnessEvaluator:
    """Create a sample fitness evaluator."""
    return FitnessEvaluator(fitness_functions=sample_fitness_functions, fitness_kwargs=sample_fitness_args)

  def test_init_success(self, sample_fitness_evaluator: FitnessEvaluator) -> None:
    """Test successful FitnessEvaluator initialization."""
    evaluator = sample_fitness_evaluator
    assert len(evaluator.fitness_functions) == 3
    assert evaluator.combine_func.func == "sum"

  def test_init_empty_fitness_functions(self) -> None:
    """Test FitnessEvaluator raises ValueError for empty fitness functions."""
    with pytest.raises(ValueError, match="At least one fitness function must be provided"):
      FitnessEvaluator(fitness_functions=())

  def test_get_functions_by_type_protein(
    self,
    sample_fitness_evaluator: FitnessEvaluator,
  ) -> None:
    """Test getting protein-type fitness functions."""
    protein_funcs = sample_fitness_evaluator.get_functions_by_type("protein")
    assert len(protein_funcs) == 2
    assert all(f.input_type == "protein" for f in protein_funcs)

  def test_get_functions_by_type_nucleotide(
    self,
    sample_fitness_evaluator: FitnessEvaluator,
  ) -> None:
    """Test getting nucleotide-type fitness functions."""
    nucleotide_funcs = sample_fitness_evaluator.get_functions_by_type("nucleotide")
    assert len(nucleotide_funcs) == 1
    assert all(f.input_type == "nucleotide" for f in nucleotide_funcs)

  def test_pytree_registration(self, sample_fitness_evaluator: FitnessEvaluator) -> None:
    """Test that FitnessEvaluator is properly registered as a PyTree."""
    evaluator = sample_fitness_evaluator

    # Test that it can be used in JAX transformations
    def process_evaluator(e: FitnessEvaluator) -> int:
      return len(e.fitness_functions)

    jitted_process = jax.jit(process_evaluator)
    result = jitted_process(evaluator)
    chex.assert_trees_all_close(result, 3)

    # Test tree flatten/unflatten
    leaves, treedef = jax.tree_util.tree_flatten(evaluator)
    unflattened_evaluator = jax.tree_util.tree_unflatten(treedef, leaves)

    assert len(evaluator.fitness_functions) == len(unflattened_evaluator.fitness_functions)
    assert evaluator.combine_func.func == unflattened_evaluator.combine_func.func

  def test_get_score_fns(self, sample_fitness_evaluator: FitnessEvaluator) -> None:
    """Test get_score_fns method."""
    mock_registry = Mock(spec=FitnessRegistry)
    mock_item = Mock()
    mock_item.return_value = make_mock_fitness_function()
    mock_registry.__contains__.return_value = True
    mock_registry.get.return_value = mock_item

 

    score_fns = sample_fitness_evaluator.get_score_fns(mock_registry)
    assert len(score_fns) == 3
    assert all(callable(fn) for fn in score_fns)

  def test_combine_success(self, sample_fitness_evaluator: FitnessEvaluator) -> None:
    """Test successful combine method."""
    mock_registry = Mock(spec=CombineRegistry)
    mock_item = Mock()
    mock_item.return_value = make_mock_combine_function()
    mock_registry.__contains__.return_value = True
    mock_registry.get.return_value = mock_item

    evaluator = sample_fitness_evaluator
    combine_fn = evaluator.combine(mock_registry)

    mock_registry.get.assert_called_once_with("sum")
    assert callable(combine_fn)

  def test_combine_missing_function(self, sample_fitness_evaluator: FitnessEvaluator) -> None:
    """Test combine method raises ValueError for missing function."""
    mock_registry = Mock(spec=CombineRegistry)
    mock_registry.__contains__.return_value = False

    evaluator = sample_fitness_evaluator
    with pytest.raises(ValueError, match="Combine function sum is not registered"):
      evaluator.combine(mock_registry)
