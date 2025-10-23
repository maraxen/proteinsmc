"""Tests for MCMC sampling loop body."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from chex import assert_shape

from proteinsmc.models.fitness import StackedFitness
from proteinsmc.models.sampler_base import SamplerState
from proteinsmc.models.types import EvoSequence
from proteinsmc.sampling.mcmc import run_mcmc_loop


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


@pytest.fixture
def simple_mutation_fn():
  """Create a simple mutation function for testing.
  
  Args:
      None
  
  Returns:
      Callable: A mutation function that adds random noise.
  
  Raises:
      None
  
  Example:
      >>> fn = simple_mutation_fn()
      >>> mutated = fn(key, sequence)
  
  """

  def mutation_fn(key: jax.Array, sequence: EvoSequence) -> EvoSequence:
    """Simple mutation that adds random noise."""
    noise = jax.random.randint(key, sequence.shape, 0, 2)
    return (sequence + noise) % 20

  return mutation_fn


class TestRunMCMCLoop:
  """Test the run_mcmc_loop function."""

  def test_loop_executes(self, simple_fitness_fn, simple_mutation_fn) -> None:
    """Test that the MCMC loop executes successfully.
    
    Args:
        simple_fitness_fn: Fixture providing a simple fitness function.
        simple_mutation_fn: Fixture providing a simple mutation function.
    
    Returns:
        None
    
    Raises:
        AssertionError: If loop execution fails.
    
    Example:
        >>> test_loop_executes(simple_fitness_fn, simple_mutation_fn)
    
    """
    key = jax.random.PRNGKey(42)
    sequence = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int8)
    fitness = simple_fitness_fn(key, sequence)

    initial_state = SamplerState(
      sequence=sequence,
      key=key,
      step=jnp.array(0),
    )

    final_state, metrics = run_mcmc_loop(
      num_samples=5,
      initial_state=initial_state,
      fitness_fn=simple_fitness_fn,
      mutation_fn=simple_mutation_fn,
    )

    assert final_state.step == 5
    assert_shape(final_state.sequence, (5,))
    assert isinstance(metrics, dict)

  def test_loop_with_io_callback(self, simple_fitness_fn, simple_mutation_fn) -> None:
    """Test MCMC loop with I/O callback.
    
    Args:
        simple_fitness_fn: Fixture providing a simple fitness function.
        simple_mutation_fn: Fixture providing a simple mutation function.
    
    Returns:
        None
    
    Raises:
        AssertionError: If loop with callback fails.
    
    Example:
        >>> test_loop_with_io_callback(simple_fitness_fn, simple_mutation_fn)
    
    """
    key = jax.random.PRNGKey(42)
    sequence = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int8)
    fitness = simple_fitness_fn(key, sequence)

    initial_state = SamplerState(
      sequence=sequence,
      key=key,
      step=jnp.array(0),
    )

    callback_data = []

    def io_callback(data):
      """Collect data from callback."""
      callback_data.append(data)

    final_state, _ = run_mcmc_loop(
      num_samples=3,
      initial_state=initial_state,
      fitness_fn=simple_fitness_fn,
      mutation_fn=simple_mutation_fn,
      io_callback=io_callback,
    )

    assert final_state.step == 3
    # Callback should be invoked for each step
    assert len(callback_data) == 3

  def test_loop_state_progression(self, simple_fitness_fn, simple_mutation_fn) -> None:
    """Test that state progresses correctly through MCMC loop.
    
    Args:
        simple_fitness_fn: Fixture providing a simple fitness function.
        simple_mutation_fn: Fixture providing a simple mutation function.
    
    Returns:
        None
    
    Raises:
        AssertionError: If state progression is incorrect.
    
    Example:
        >>> test_loop_state_progression(simple_fitness_fn, simple_mutation_fn)
    
    """
    key = jax.random.PRNGKey(42)
    sequence = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int8)
    fitness = simple_fitness_fn(key, sequence)

    initial_state = SamplerState(
      sequence=sequence,
      key=key,
      step=jnp.array(0),
    )

    # Run for different numbers of steps
    for num_samples in [1, 5, 10]:
      final_state, _ = run_mcmc_loop(
        num_samples=num_samples,
        initial_state=initial_state,
        fitness_fn=simple_fitness_fn,
        mutation_fn=simple_mutation_fn,
      )
      assert final_state.step == num_samples

  def test_loop_returns_empty_metrics(self, simple_fitness_fn, simple_mutation_fn) -> None:
    """Test that MCMC loop returns empty metrics dictionary.
    
    Args:
        simple_fitness_fn: Fixture providing a simple fitness function.
        simple_mutation_fn: Fixture providing a simple mutation function.
    
    Returns:
        None
    
    Raises:
        AssertionError: If metrics are not empty dict.
    
    Example:
        >>> test_loop_returns_empty_metrics(simple_fitness_fn, simple_mutation_fn)
    
    """
    key = jax.random.PRNGKey(42)
    sequence = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int8)
    fitness = simple_fitness_fn(key, sequence)

    initial_state = SamplerState(
      sequence=sequence,
      fitness=fitness,
      key=key,
      step=jnp.array(0),
    )

    _, metrics = run_mcmc_loop(
      num_samples=5,
      initial_state=initial_state,
      fitness_fn=simple_fitness_fn,
      mutation_fn=simple_mutation_fn,
    )

    assert metrics == {}
    assert isinstance(metrics, dict)
