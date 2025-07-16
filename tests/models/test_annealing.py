from __future__ import annotations
from typing import Any, Callable
import chex
import jax
import pytest
from jaxtyping import Float, Int

"""Tests for annealing schedule data structures."""



import jax.numpy as jnp

from proteinsmc.models.annealing import (
  AnnealingRegistryItem,
  AnnealingScheduleConfig,
  AnnealingScheduleRegistry,
)

# Define type aliases for clarity in mock functions
CurrentStepInt = Int[jax.Array, "current_step"]
ScheduleLenInt = Int[jax.Array, "schedule_len"]
MaxBetaFloat = Float[jax.Array, "max_beta"]
CurrentBetaFloat = Float[jax.Array, "current_beta"]


# Mock annealing schedule functions for testing
def linear_schedule(
  p: CurrentStepInt, n_steps: ScheduleLenInt, beta_max: MaxBetaFloat
) -> CurrentBetaFloat:
  """A mock linear schedule."""
  return beta_max * (p - 1) / (n_steps - 1)


def constant_schedule(
  _p: CurrentStepInt, _n_steps: ScheduleLenInt, beta_max: MaxBetaFloat
) -> CurrentBetaFloat:
  """A mock constant schedule."""
  return beta_max


@pytest.fixture
def linear_schedule_item() -> AnnealingRegistryItem:
  """Fixture for a linear annealing schedule registry item."""
  return AnnealingRegistryItem(
    method_factory=lambda: linear_schedule, name="linear"
  )


@pytest.fixture
def constant_schedule_item() -> AnnealingRegistryItem:
  """Fixture for a constant annealing schedule registry item."""
  return AnnealingRegistryItem(
    method_factory=lambda: constant_schedule, name="constant"
  )


@pytest.fixture
def sample_registry(
  linear_schedule_item: AnnealingRegistryItem,
  constant_schedule_item: AnnealingRegistryItem,
) -> AnnealingScheduleRegistry:
  """Fixture for a sample AnnealingScheduleRegistry."""
  return AnnealingScheduleRegistry(
    items={"linear": linear_schedule_item, "constant": constant_schedule_item}
  )


def test_registry_init_success(sample_registry: AnnealingScheduleRegistry):
  """Test that AnnealingScheduleRegistry initializes correctly with valid items."""
  assert "linear" in sample_registry
  assert "constant" in sample_registry
  assert len(sample_registry.items) == 2


def test_registry_init_type_error():
  """Test that AnnealingScheduleRegistry raises TypeError for invalid item types."""
  with pytest.raises(TypeError, match="All items in AnnealingScheduleRegistry must be"):
    AnnealingScheduleRegistry(items={"invalid": "not_an_item"})  # type: ignore


def test_registry_get_item(
  sample_registry: AnnealingScheduleRegistry,
  linear_schedule_item: AnnealingRegistryItem,
):
  """Test retrieving an item from the registry."""
  item = sample_registry.get("linear")
  assert item == linear_schedule_item
  assert item.name == "linear"


def test_registry_get_nonexistent_item(sample_registry: AnnealingScheduleRegistry):
  """Test that getting a non-existent item raises a KeyError."""
  with pytest.raises(KeyError):
    sample_registry.get("nonexistent")


def test_registry_contains_item(sample_registry: AnnealingScheduleRegistry):
  """Test the `in` operator for checking item existence."""
  assert "linear" in sample_registry
  assert "nonexistent" not in sample_registry


def test_annealing_schedule_config_call(sample_registry: AnnealingScheduleRegistry):
  """Test calling an AnnealingScheduleConfig to get a schedule function."""
  config = AnnealingScheduleConfig(
    schedule_fn="linear", beta_max=1.0, n_steps=10
  )
  schedule_func = config(sample_registry)
  assert isinstance(schedule_func, Callable)

  # Test the retrieved function
  beta = schedule_func(
    p=jnp.array(5), n_steps=jnp.array(10), beta_max=jnp.array(1.0)
  )
  chex.assert_trees_all_close(beta, 4.0 / 9.0)


def test_annealing_schedule_config_call_invalid(
  sample_registry: AnnealingScheduleRegistry,
):
  """Test that calling a config with an unregistered schedule raises ValueError."""
  config = AnnealingScheduleConfig(
    schedule_fn="nonexistent", beta_max=1.0, n_steps=10
  )
  with pytest.raises(ValueError, match="Annealing schedule 'nonexistent' is not registered."):
    config(sample_registry)


def test_annealing_schedule_config_pytree_registration():
  """Test that AnnealingScheduleConfig is registered as a PyTree."""
  config = AnnealingScheduleConfig(
    schedule_fn="linear",
    beta_max=1.0,
    n_steps=10,
    schedule_args={"extra": 5},
  )

  def process_config(c: AnnealingScheduleConfig) -> float:
    return c.beta_max * c.n_steps

  jitted_process = jax.jit(process_config)
  result = jitted_process(config)

  chex.assert_trees_all_close(result, 10.0)

  # Test tree flatten/unflatten
  leaves, treedef = jax.tree_util.tree_flatten(config)
  unflattened_config = jax.tree_util.tree_unflatten(treedef, leaves)

  assert config == unflattened_config
  assert leaves[0] == "linear"
  assert leaves[1] == 1.0
  assert leaves[2] == 10
  assert leaves[3] == {"extra": 5}