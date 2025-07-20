import jax
import pytest

from .conftest import valid_config_kwargs
from proteinsmc.models.hmc import HMCConfig, HMCState
from proteinsmc.models.sampler_base import BaseSamplerConfig

"""Tests for HMC sampler data structures."""

import jax.numpy as jnp
import jax.random as jr


class TestHMCState:
  """Test the HMCState class."""

  def test_hmc_state_creation(self) -> None:
    """Test that HMCState can be created with valid inputs.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the state attributes do not match expected values.

    Example:
        >>> test_hmc_state_creation()

    """
    key = jr.PRNGKey(42)
    samples = jnp.array([1, 2, 3, 4, 5])
    fitness = jnp.array(0.85)

    state = HMCState(samples=samples, fitness=fitness, key=key)

    assert jnp.array_equal(state.samples, samples)
    assert jnp.array_equal(state.fitness, fitness)
    assert jnp.array_equal(state.key, key)

  def test_hmc_state_pytree_properties(self) -> None:
    """Test that HMCState behaves as a proper PyTree.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If PyTree operations fail.

    Example:
        >>> test_hmc_state_pytree_properties()

    """
    key = jr.PRNGKey(42)
    samples = jnp.array([1, 2, 3, 4, 5])
    fitness = jnp.array(0.85)

    state = HMCState(samples=samples, fitness=fitness, key=key)

    # Test that it can be flattened and unflattened
    flat, tree_def = jax.tree_util.tree_flatten(state)
    reconstructed = jax.tree_util.tree_unflatten(tree_def, flat)

    assert jnp.array_equal(reconstructed.samples, state.samples)
    assert jnp.array_equal(reconstructed.fitness, state.fitness)
    assert jnp.array_equal(reconstructed.key, state.key)

  def test_hmc_state_immutability(self) -> None:
    """Test that HMCState supports immutable updates.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If immutable updates fail.

    Example:
        >>> test_hmc_state_immutability()

    """
    key = jr.PRNGKey(42)
    samples = jnp.array([1, 2, 3, 4, 5])
    fitness = jnp.array(0.85)

    state = HMCState(samples=samples, fitness=fitness, key=key)

    # Test immutable update
    new_fitness = jnp.array(0.95)
    new_state = state.replace(fitness=new_fitness)

    # Original state should be unchanged
    assert jnp.array_equal(state.fitness, fitness)
    # New state should have updated fitness
    assert jnp.array_equal(new_state.fitness, new_fitness)
    # Other attributes should remain the same
    assert jnp.array_equal(new_state.samples, samples)
    assert jnp.array_equal(new_state.key, key)


class TestHMCConfig:
  """Test the HMCConfig class."""

  def test_hmc_config_default_values(self, valid_config_kwargs) -> None:
    """Test that HMCConfig has correct default values.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If default values do not match expected values.

    Example:
        >>> test_hmc_config_default_values()

    """
    config = HMCConfig(**valid_config_kwargs)
    print(config)
    assert config.step_size == 0.1
    assert config.num_leapfrog_steps == 10

  def test_hmc_config_custom_values(self, valid_config_kwargs) -> None:
    """Test that HMCConfig accepts custom values.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If custom values are not set correctly.

    Example:
        >>> test_hmc_config_custom_values()

    """
    step_size = 0.05
    num_leapfrog_steps = 20

    config = HMCConfig(
      **valid_config_kwargs,
      step_size=step_size,
      num_leapfrog_steps=num_leapfrog_steps,
    )

    assert config.step_size == step_size
    assert config.num_leapfrog_steps == num_leapfrog_steps

  def test_hmc_config_inheritance(self) -> None:
    """Test that HMCConfig inherits from BaseSamplerConfig.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If inheritance is not working properly.

    Example:
        >>> test_hmc_config_inheritance()

    """
    config = HMCConfig()

    assert isinstance(config, BaseSamplerConfig)

  def test_hmc_config_field_types(self) -> None:
    """Test that HMCConfig field types are correct.

    Args:
        None

    Returns:
        None

    Raises:
        TypeError: If field types are incorrect.

    Example:
        >>> test_hmc_config_field_types()

    """
    config = HMCConfig(step_size=0.05, num_leapfrog_steps=15)

    if not isinstance(config.step_size, float):
      raise TypeError(f"Expected float, but got {type(config.step_size)}")
    if not isinstance(config.num_leapfrog_steps, int):
      raise TypeError(f"Expected int, but got {type(config.num_leapfrog_steps)}")

  def test_hmc_config_invalid_types(self) -> None:
    """Test that HMCConfig raises errors for invalid types.

    Args:
        None

    Returns:
        None

    Raises:
        TypeError: If invalid types are not properly handled.

    Example:
        >>> test_hmc_config_invalid_types()

    """
    # Test that string step_size raises TypeError when used
    with pytest.raises(TypeError):
      config = HMCConfig(step_size="invalid")
      # Force type checking by attempting to use in computation
      _ = config.step_size + 1.0

    # Test that float num_leapfrog_steps raises TypeError when used as int
    with pytest.raises(TypeError):
      config = HMCConfig(num_leapfrog_steps=10.5)
      # Force type checking by attempting to use in range
      _ = range(config.num_leapfrog_steps)
