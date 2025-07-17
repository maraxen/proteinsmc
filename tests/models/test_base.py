"""Tests for base model classes and protocols."""

from __future__ import annotations

from abc import ABC
from typing import Any, Callable
from unittest.mock import Mock

import chex
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Float, Int

from proteinsmc.models.registry_base import (
  RegisteredFunction,
  Registry,
  RegistryItem,
)
from proteinsmc.models.sampler_base import BaseSamplerConfig
from proteinsmc.models.fitness import FitnessEvaluator, FitnessFunction
from proteinsmc.models.memory import MemoryConfig


# Mock classes for testing abstract base classes
class MockRegistryItem(RegistryItem):
  """Mock implementation of RegistryItem for testing."""

  def __call__(self, *args: Any, **kwargs: Any) -> Callable:  # noqa: ANN401
    """Mock call method."""
    return self.method_factory(*args, **kwargs)


class MockRegistry(Registry):
  """Mock implementation of Registry for testing."""


# Mock fitness evaluator for BaseSamplerConfig
@pytest.fixture
def mock_fitness_evaluator() -> FitnessEvaluator:
  """Create a mock FitnessEvaluator for testing."""
  mock_fitness_func = FitnessFunction(func="mock_fitness")
  return FitnessEvaluator(fitness_functions=(mock_fitness_func,))


@pytest.fixture
def mock_memory_config() -> MemoryConfig:
  """Create a mock MemoryConfig for testing."""
  return MemoryConfig()


@pytest.fixture
def valid_sampler_config(
  mock_fitness_evaluator: FitnessEvaluator,
  mock_memory_config: MemoryConfig,
) -> BaseSamplerConfig:
  """Create a valid BaseSamplerConfig for testing."""
  return BaseSamplerConfig(
    seed_sequence="MKLLVL",
    generations=10,
    n_states=100,
    mutation_rate=0.1,
    diversification_ratio=0.2,
    sequence_type="protein",
    fitness_evaluator=mock_fitness_evaluator,
    memory_config=mock_memory_config,
  )


class TestRegistryItem:
  """Test cases for RegistryItem base class."""

  def test_init_success(self) -> None:
    """Test successful initialization of RegistryItem."""
    def mock_factory() -> Callable:
      return lambda x: x

    item = MockRegistryItem(method_factory=mock_factory, name="test_item")
    assert item.name == "test_item"
    assert callable(item.method_factory)

  def test_init_invalid_name_type(self) -> None:
    """Test that RegistryItem raises TypeError for invalid name type."""
    def mock_factory() -> Callable:
      return lambda x: x

    with pytest.raises(TypeError, match="name must be a string"):
      MockRegistryItem(method_factory=mock_factory, name=123)  # type: ignore

  def test_init_invalid_factory_type(self) -> None:
    """Test that RegistryItem raises TypeError for non-callable factory."""
    with pytest.raises(TypeError, match="Method factory .* is not callable"):
      MockRegistryItem(method_factory="not_callable", name="test")  # type: ignore


class TestRegistry:
  """Test cases for Registry base class."""

  @pytest.fixture
  def sample_item(self) -> MockRegistryItem:
    """Create a sample registry item."""
    def mock_factory() -> Callable:
      return lambda x: x

    return MockRegistryItem(method_factory=mock_factory, name="sample")

  @pytest.fixture
  def sample_registry(self, sample_item: MockRegistryItem) -> MockRegistry:
    """Create a sample registry with one item."""
    return MockRegistry(items={"sample": sample_item})

  def test_init_success(self, sample_registry: MockRegistry) -> None:
    """Test successful Registry initialization."""
    assert "sample" in sample_registry
    assert len(sample_registry.items) == 1

  def test_init_invalid_items_type(self) -> None:
    """Test Registry raises TypeError for invalid items type."""
    with pytest.raises(TypeError, match="items must be a dictionary"):
      MockRegistry(items="not_a_dict")  # type: ignore

  def test_init_invalid_item_type(self) -> None:
    """Test Registry raises TypeError for invalid item types."""
    with pytest.raises(TypeError, match="All items must be instances of RegistryItem"):
      MockRegistry(items={"invalid": "not_an_item"})  # type: ignore

  def test_contains(self, sample_registry: MockRegistry) -> None:
    """Test the __contains__ method."""
    assert "sample" in sample_registry
    assert "nonexistent" not in sample_registry

  def test_getitem_success(
    self,
    sample_registry: MockRegistry,
    sample_item: MockRegistryItem,
  ) -> None:
    """Test successful item retrieval."""
    retrieved_item = sample_registry["sample"]
    assert retrieved_item == sample_item

  def test_getitem_missing(self, sample_registry: MockRegistry) -> None:
    """Test KeyError for missing item."""
    with pytest.raises(KeyError, match="Item nonexistent is not registered"):
      sample_registry["nonexistent"]

  def test_register_new_item(self, sample_registry: MockRegistry) -> None:
    """Test registering a new item."""
    def new_factory() -> Callable:
      return lambda x: x * 2

    new_item = MockRegistryItem(method_factory=new_factory, name="new_item")
    sample_registry.register(new_item)

    assert "new_item" in sample_registry
    assert sample_registry["new_item"] == new_item

  def test_register_duplicate_item(
    self,
    sample_registry: MockRegistry,
    sample_item: MockRegistryItem,
  ) -> None:
    """Test that registering duplicate item raises ValueError."""
    with pytest.raises(ValueError, match="Item sample is already registered"):
      sample_registry.register(sample_item)

  def test_get_success(
    self,
    sample_registry: MockRegistry,
    sample_item: MockRegistryItem,
  ) -> None:
    """Test successful get method."""
    retrieved_item = sample_registry.get("sample")
    assert retrieved_item == sample_item

  def test_get_missing(self, sample_registry: MockRegistry) -> None:
    """Test get method raises KeyError for missing item."""
    with pytest.raises(KeyError, match="Item nonexistent is not registered"):
      sample_registry.get("nonexistent")


class TestRegisteredFunction:
  """Test cases for RegisteredFunction."""

  def test_init_success(self) -> None:
    """Test successful RegisteredFunction initialization."""
    rf = RegisteredFunction(
      func="test_func",
      required_args=(int, str),
      required_kwargs={"param": float},
    )
    assert rf.func == "test_func"
    assert rf.required_args == (int, str)
    assert rf.required_kwargs == {"param": float}

  def test_init_invalid_func_type(self) -> None:
    """Test RegisteredFunction raises TypeError for invalid func type."""
    with pytest.raises(TypeError, match="Registered function .* is not a string"):
      RegisteredFunction(func=123)  # type: ignore

  def test_init_invalid_context_tuple_type(self) -> None:
    """Test RegisteredFunction raises TypeError for invalid context_tuple."""
    with pytest.raises(TypeError, match="context_tuple must be a tuple"):
      RegisteredFunction(func="test", context_tuple="not_tuple")  # type: ignore

  def test_init_invalid_required_args_type(self) -> None:
    """Test RegisteredFunction raises TypeError for invalid required_args."""
    with pytest.raises(TypeError, match="required_args must be a tuple"):
      RegisteredFunction(func="test", required_args="not_tuple")  # type: ignore

  def test_init_invalid_required_args_content(self) -> None:
    """Test RegisteredFunction raises TypeError for non-type required_args."""
    with pytest.raises(TypeError, match="All required_args must be types"):
      RegisteredFunction(func="test", required_args=("not_type",))  # type: ignore

  def test_init_invalid_required_kwargs_type(self) -> None:
    """Test RegisteredFunction raises TypeError for invalid required_kwargs."""
    with pytest.raises(TypeError, match="required_kwargs must be a dictionary"):
      RegisteredFunction(func="test", required_kwargs="not_dict")  # type: ignore

  def test_init_invalid_required_kwargs_values(self) -> None:
    """Test RegisteredFunction raises TypeError for non-type kwargs values."""
    with pytest.raises(TypeError, match="All required_kwargs values must be types"):
      RegisteredFunction(func="test", required_kwargs={"param": "not_type"})  # type: ignore

  def test_call_success(self) -> None:
    """Test successful call of RegisteredFunction."""
    mock_registry = Mock(spec=Registry)
    mock_item = Mock()
    mock_item.return_value = lambda x: x * 2
    mock_registry.__contains__.return_value = True
    mock_registry.get.return_value = mock_item

    rf = RegisteredFunction(func="test_func")
    result_func = rf(mock_registry)

    mock_registry.get.assert_called_once_with("test_func")
    assert callable(result_func)

  def test_call_invalid_registry_type(self) -> None:
    """Test RegisteredFunction call raises TypeError for invalid registry."""
    rf = RegisteredFunction(func="test_func")
    with pytest.raises(TypeError, match="Expected registry to be an instance of Registry"):
      rf("not_registry")  # type: ignore

  def test_call_missing_function(self) -> None:
    """Test RegisteredFunction call raises ValueError for missing function."""
    mock_registry = Mock(spec=Registry)
    mock_registry.__contains__.return_value = False

    rf = RegisteredFunction(func="missing_func")
    with pytest.raises(ValueError, match="Function missing_func is not registered"):
      rf(mock_registry)

  def test_pytree_registration(self) -> None:
    """Test that RegisteredFunction is properly registered as a PyTree."""
    rf = RegisteredFunction(
      func="test_func",
      required_args=(int,),
      required_kwargs={"param": float},
    )

    # Test that it can be used in JAX transformations
    def process_rf(rf_input: RegisteredFunction) -> str:
      return rf_input.func

    jitted_process = jax.jit(process_rf)
    result = jitted_process(rf)
    assert result == "test_func"

    # Test tree flatten/unflatten
    leaves, treedef = jax.tree_util.tree_flatten(rf)
    unflattened_rf = jax.tree_util.tree_unflatten(treedef, leaves)

    assert rf.func == unflattened_rf.func
    assert rf.required_args == unflattened_rf.required_args
    assert rf.required_kwargs == unflattened_rf.required_kwargs


class TestBaseSamplerConfig:
  """Test cases for BaseSamplerConfig."""

  def test_init_success(self, valid_sampler_config: BaseSamplerConfig) -> None:
    """Test successful BaseSamplerConfig initialization."""
    config = valid_sampler_config
    assert config.seed_sequence == "MKLLVL"
    assert config.generations == 10
    assert config.n_states == 100
    assert config.mutation_rate == 0.1
    assert config.diversification_ratio == 0.2
    assert config.sequence_type == "protein"

  def test_validation_negative_n_states(
    self,
    mock_fitness_evaluator: FitnessEvaluator,
    mock_memory_config: MemoryConfig,
  ) -> None:
    """Test validation fails for negative n_states."""
    with pytest.raises(ValueError, match="n_states must be positive"):
      BaseSamplerConfig(
        seed_sequence="MKLLVL",
        generations=10,
        n_states=-1,
        mutation_rate=0.1,
        diversification_ratio=0.2,
        sequence_type="protein",
        fitness_evaluator=mock_fitness_evaluator,
        memory_config=mock_memory_config,
      )

  def test_validation_negative_generations(
    self,
    mock_fitness_evaluator: FitnessEvaluator,
    mock_memory_config: MemoryConfig,
  ) -> None:
    """Test validation fails for negative generations."""
    with pytest.raises(ValueError, match="generations must be positive"):
      BaseSamplerConfig(
        seed_sequence="MKLLVL",
        generations=-1,
        n_states=100,
        mutation_rate=0.1,
        diversification_ratio=0.2,
        sequence_type="protein",
        fitness_evaluator=mock_fitness_evaluator,
        memory_config=mock_memory_config,
      )

  def test_validation_invalid_mutation_rate(
    self,
    mock_fitness_evaluator: FitnessEvaluator,
    mock_memory_config: MemoryConfig,
  ) -> None:
    """Test validation fails for invalid mutation_rate."""
    with pytest.raises(ValueError, match="mutation_rate must be in \\[0.0, 1.0\\]"):
      BaseSamplerConfig(
        seed_sequence="MKLLVL",
        generations=10,
        n_states=100,
        mutation_rate=1.5,
        diversification_ratio=0.2,
        sequence_type="protein",
        fitness_evaluator=mock_fitness_evaluator,
        memory_config=mock_memory_config,
      )

  def test_validation_invalid_diversification_ratio(
    self,
    mock_fitness_evaluator: FitnessEvaluator,
    mock_memory_config: MemoryConfig,
  ) -> None:
    """Test validation fails for invalid diversification_ratio."""
    with pytest.raises(ValueError, match="diversification_ratio must be in \\[0.0, 1.0\\]"):
      BaseSamplerConfig(
        seed_sequence="MKLLVL",
        generations=10,
        n_states=100,
        mutation_rate=0.1,
        diversification_ratio=1.5,
        sequence_type="protein",
        fitness_evaluator=mock_fitness_evaluator,
        memory_config=mock_memory_config,
      )

  def test_validation_invalid_sequence_type(
    self,
    mock_fitness_evaluator: FitnessEvaluator,
    mock_memory_config: MemoryConfig,
  ) -> None:
    """Test validation fails for invalid sequence_type."""
    with pytest.raises(ValueError, match="sequence_type must be 'protein' or 'nucleotide'"):
      BaseSamplerConfig(
        seed_sequence="MKLLVL",
        generations=10,
        n_states=100,
        mutation_rate=0.1,
        diversification_ratio=0.2,
        sequence_type="invalid",  # type: ignore
        fitness_evaluator=mock_fitness_evaluator,
        memory_config=mock_memory_config,
      )

  def test_pytree_registration(self, valid_sampler_config: BaseSamplerConfig) -> None:
    """Test that BaseSamplerConfig is properly registered as a PyTree."""
    config = valid_sampler_config

    # Test that it can be used in JAX transformations
    def process_config(c: BaseSamplerConfig) -> int:
      return c.generations

    jitted_process = jax.jit(process_config)
    result = jitted_process(config)
    chex.assert_trees_all_close(result, 10)

    # Test tree flatten/unflatten
    leaves, treedef = jax.tree_util.tree_flatten(config)
    unflattened_config = jax.tree_util.tree_unflatten(treedef, leaves)

    assert config.seed_sequence == unflattened_config.seed_sequence
    assert config.generations == unflattened_config.generations
    assert config.n_states == unflattened_config.n_states
    assert config.mutation_rate == unflattened_config.mutation_rate
    assert config.diversification_ratio == unflattened_config.diversification_ratio
    assert config.sequence_type == unflattened_config.sequence_type

  def test_additional_config_fields(self, valid_sampler_config: BaseSamplerConfig) -> None:
    """Test the additional_config_fields property."""
    config = valid_sampler_config
    fields = config.additional_config_fields
    assert isinstance(fields, dict)
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in fields.items())
