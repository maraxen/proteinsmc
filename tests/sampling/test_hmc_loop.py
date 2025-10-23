"""Tests for HMC sampling loop body."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from chex import assert_shape

from proteinsmc.models.fitness import StackedFitness
from proteinsmc.models.sampler_base import SamplerState
from proteinsmc.models.types import EvoSequence
from proteinsmc.sampling.hmc import run_hmc_loop


@pytest.fixture
def simple_fitness_fn():
  """Create a simple fitness function for testing.
  
  Args:
      None
  
  Returns:
      Callable: A fitness function that returns sum of sequence values.
  
  Raises:
      None
  
  Example:
      >>> fn = simple_fitness_fn()
      >>> fitness = fn(key, sequence)
  
  """

  def fitness_fn(key: jax.Array, sequence: EvoSequence, _context=None) -> StackedFitness:
    """Simple fitness function that sums sequence values."""
    return jnp.stack([jnp.sum(sequence).astype(jnp.float32), jnp.sum(sequence).astype(jnp.float32)])

  return fitness_fn


class TestRunHMCLoop:
  """Test the run_hmc_loop function."""

  @pytest.mark.skip(reason="Tests need to be updated for new HMC loop implementation.")
  def test_loop_executes(self, simple_fitness_fn) -> None:
    """Test that the HMC loop executes successfully.
    
    Args:
        simple_fitness_fn: Fixture providing a simple fitness function.
    
    Returns:
        None
    
    Raises:
        AssertionError: If loop execution fails.
    
    Example:
        >>> test_loop_executes(simple_fitness_fn)
    
    """
    key = jax.random.PRNGKey(42)
    sequence = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32)
    fitness = simple_fitness_fn(key, sequence)

    initial_state = SamplerState(
      sequence=sequence,
      key=key,
      step=jnp.array(0),
    )

    final_state, metrics = run_hmc_loop(
      num_samples=5,
      initial_state=initial_state,
      fitness_fn=simple_fitness_fn,
    )

    assert final_state.step == 5
    assert_shape(final_state.sequence, (5,))
    assert isinstance(metrics, dict)

  @pytest.mark.skip(reason="Tests need to be updated for new HMC loop implementation.")
  def test_loop_with_io_callback(self, simple_fitness_fn) -> None:
    """Test HMC loop with I/O callback.
    
    Args:
        simple_fitness_fn: Fixture providing a simple fitness function.
    
    Returns:
        None
    
    Raises:
        AssertionError: If loop with callback fails.
    
    Example:
        >>> test_loop_with_io_callback(simple_fitness_fn)
    
    """
    key = jax.random.PRNGKey(42)
    sequence = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32)

    initial_state = SamplerState(
      sequence=sequence,
      key=key,
      step=jnp.array(0),
    )

    callback_data = []

    def io_callback(data):
      """Collect data from callback."""
      callback_data.append(data)

    final_state, _ = run_hmc_loop(
      num_samples=3,
      initial_state=initial_state,
      fitness_fn=simple_fitness_fn,
      io_callback=io_callback,
    )

    assert final_state.step == 3
    assert len(callback_data) == 3

  @pytest.mark.skip(reason="Tests need to be updated for new HMC loop implementation.")
  def test_loop_state_progression(self, simple_fitness_fn) -> None:
    """Test that state progresses correctly through HMC loop.
    
    Args:
        simple_fitness_fn: Fixture providing a simple fitness function.
    
    Returns:
        None
    
    Raises:
        AssertionError: If state progression is incorrect.
    
    Example:
        >>> test_loop_state_progression(simple_fitness_fn)
    
    """
    key = jax.random.PRNGKey(42)
    sequence = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32)
    fitness = simple_fitness_fn(key, sequence)

    initial_state = SamplerState(
      sequence=sequence,
      key=key,
      step=jnp.array(0),
    )

    for num_samples in [1, 5, 10]:
      final_state, _ = run_hmc_loop(
        num_samples=num_samples,
        initial_state=initial_state,
        fitness_fn=simple_fitness_fn,
      )
      assert final_state.step == num_samples

  @pytest.mark.skip(reason="Tests need to be updated for new HMC loop implementation.")
  def test_loop_returns_empty_metrics(self, simple_fitness_fn) -> None:
    """Test that HMC loop returns empty metrics dictionary.
    
    Args:
        simple_fitness_fn: Fixture providing a simple fitness function.
    
    Returns:
        None
    
    Raises:
        AssertionError: If metrics are not empty dict.
    
    Example:
        >>> test_loop_returns_empty_metrics(simple_fitness_fn)
    
    """
    key = jax.random.PRNGKey(42)
    sequence = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32)
    fitness = simple_fitness_fn(key, sequence)

    initial_state = SamplerState(
      sequence=sequence,
      key=key,
      step=jnp.array(0),
    )

    _, metrics = run_hmc_loop(
      num_samples=5,
      initial_state=initial_state,
      fitness_fn=simple_fitness_fn,
    )

    assert metrics == {}
    assert isinstance(metrics, dict)

  @pytest.mark.skip(reason="Tests need to be updated for new HMC loop implementation.")
  def test_loop_ignores_mutation_fn(self, simple_fitness_fn) -> None:
    """Test that HMC loop ignores mutation_fn parameter.
    
    Args:
        simple_fitness_fn: Fixture providing a simple fitness function.
    
    Returns:
        None
    
    Raises:
        AssertionError: If loop fails with mutation_fn parameter.
    
    Example:
        >>> test_loop_ignores_mutation_fn(simple_fitness_fn)
    
    """
    key = jax.random.PRNGKey(42)
    sequence = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32)
    fitness = simple_fitness_fn(key, sequence)

    initial_state = SamplerState(
      sequence=sequence,
      key=key,
      step=jnp.array(0),
    )

    # Pass a mutation_fn, which should be ignored
    def dummy_mutation_fn(key, sequence):
      return sequence

    final_state, _ = run_hmc_loop(
      num_samples=5,
      initial_state=initial_state,
      fitness_fn=simple_fitness_fn,
      _mutation_fn=dummy_mutation_fn,
    )

    assert final_state.step == 5
