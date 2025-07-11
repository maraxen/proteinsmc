"""Tests for SMC step logic and chunked processing."""


import jax
import jax.numpy as jnp
import chex
import pytest
from jax import random

from proteinsmc.sampling.smc.data_structures import MemoryConfig, SMCCarryState, SMCConfig
from proteinsmc.sampling.smc.step import smc_step
from proteinsmc.utils import AnnealingScheduleConfig, FitnessEvaluator, FitnessFunction, linear_schedule, safe_weighted_mean, chunked_mutation_step, chunked_calculate_population_fitness


def mock_fitness_fn(key, sequence, **kwargs):
  """Mock fitness function for testing."""
  return jnp.sum(sequence, axis=-1).astype(jnp.float32)


@pytest.fixture
def sample_fitness_evaluator():
  """Create a sample fitness evaluator for testing."""
  return FitnessEvaluator(
    fitness_functions=(
      FitnessFunction(
        func=mock_fitness_fn,
        input_type="protein",
        name="test_fitness",
      ),
    ),
  )


@pytest.fixture
def sample_smc_config(sample_fitness_evaluator):
  """Create a sample SMC config for testing."""
  annealing_schedule_config = AnnealingScheduleConfig(
    schedule_fn=linear_schedule,
    beta_max=1.0,
    annealing_len=50,
  )
  
  return SMCConfig(
    template_sequence="MKYN",
    population_size=100,
    n_states=20,
    generations=50,
    mutation_rate=0.1,
    diversification_ratio=0.2,
    sequence_type="protein",
    annealing_schedule_config=annealing_schedule_config,
    fitness_evaluator=sample_fitness_evaluator,
    memory_config=MemoryConfig(population_chunk_size=32),
  )


def test_safe_weighted_mean():
  """Test the safe weighted mean function."""
  metric = jnp.array([1.0, 2.0, 3.0, 4.0])
  weights = jnp.array([0.1, 0.2, 0.3, 0.4])
  valid_mask = jnp.array([True, True, False, True])
  sum_valid_w = jnp.sum(jnp.where(valid_mask, weights, 0.0))
  
  result = safe_weighted_mean(metric, weights, valid_mask, sum_valid_w)
  
  # Should compute weighted mean of valid values: (1*0.1 + 2*0.2 + 4*0.4) / (0.1+0.2+0.4)
  expected = (1.0 * 0.1 + 2.0 * 0.2 + 4.0 * 0.4) / (0.1 + 0.2 + 0.4)
  chex.assert_trees_all_close(result, expected)


def test_safe_weighted_mean_no_valid():
  """Test safe weighted mean with no valid values."""
  metric = jnp.array([1.0, 2.0, 3.0])
  weights = jnp.array([0.1, 0.2, 0.3])
  valid_mask = jnp.array([False, False, False])
  sum_valid_w = jnp.array(0.0)
  
  result = safe_weighted_mean(metric, weights, valid_mask, sum_valid_w)
  # Check that result is NaN
  assert jnp.isnan(result), f"Expected NaN, got {result}"


def test_safe_weighted_mean_type_validation():
  """Test that safe_weighted_mean validates input types."""
  with pytest.raises(TypeError, match="Expected metric to be a JAX array"):
    safe_weighted_mean([1, 2, 3], jnp.array([1, 1, 1]), jnp.array([True, True, True]), jnp.array(3.0))  # type: ignore[arg-type]
  
  with pytest.raises(TypeError, match="Expected weights to be a JAX array"):
    safe_weighted_mean(jnp.array([1, 2, 3]), [1, 1, 1], jnp.array([True, True, True]), jnp.array(3.0))  # type: ignore[arg-type]


def test_chunked_mutation_step():
  """Test chunked mutation step function."""
  key = random.PRNGKey(42)
  population = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=jnp.int8)
  mutation_rate = 0.1
  n_states = 20
  chunk_size = 2

  mutated = chunked_mutation_step(key, population, mutation_rate, n_states, chunk_size)

  chex.assert_shape(mutated, population.shape)
  chex.assert_equal(mutated.dtype, population.dtype)



def test_chunked_calculate_population_fitness(sample_fitness_evaluator):
  """Test chunked fitness evaluation function."""
  key = random.PRNGKey(42)
  population = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=jnp.int8)
  sequence_type = "protein"
  chunk_size = 2
  
  fitness_values, fitness_components = chunked_calculate_population_fitness(
    key, population, sample_fitness_evaluator, sequence_type, chunk_size
  )
  
  chex.assert_shape(fitness_values, (population.shape[0],))
  chex.assert_shape(fitness_components, (1, population.shape[0]))  # One fitness function
  chex.assert_equal(jnp.all(jnp.isfinite(fitness_values)), True)


def test_smc_step(sample_smc_config):
  """Test the main SMC step function."""
  key = random.PRNGKey(42)
  # Create population with correct shape: (population_size, sequence_length)
  population_size = 10  # Use smaller size for test efficiency
  sequence_length = len(sample_smc_config.template_sequence)
  population = random.randint(
    key, 
    (population_size, sequence_length), 
    minval=0, 
    maxval=sample_smc_config.n_states,
    dtype=jnp.int8
  )
  
  state = SMCCarryState(
    key=key,
    population=population,
    logZ_estimate=jnp.array(0.0),
    beta=jnp.array(0.5),
    step=jnp.array(0),
  )
  
  next_state, metrics = smc_step(state, sample_smc_config)
  
  # Check that state is properly updated
  chex.assert_shape(next_state.population, state.population.shape)
  chex.assert_equal(next_state.step, state.step + 1)
  chex.assert_equal(jnp.isfinite(next_state.logZ_estimate), True)
  # Check that metrics are returned
  assert "mean_combined_fitness" in metrics
  assert "max_combined_fitness" in metrics
  assert "fitness_components" in metrics
  assert "ess" in metrics
  assert "entropy" in metrics
  assert "beta" in metrics
  # Check metric shapes and values
  assert jnp.isfinite(metrics["mean_combined_fitness"]) or jnp.isnan(metrics["mean_combined_fitness"])
  assert jnp.isfinite(metrics["max_combined_fitness"]) or jnp.isnan(metrics["max_combined_fitness"])
  chex.assert_shape(metrics["fitness_components"], (1, population.shape[0]))
  chex.assert_equal(0 <= metrics["ess"] <= population.shape[0], True)


def test_smc_step_with_chunking(sample_smc_config):
  """Test SMC step with different chunk sizes."""
  key = random.PRNGKey(42)
  population = jnp.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20],
  ], dtype=jnp.int8)
  
  state = SMCCarryState(
    key=key,
    population=population,
    logZ_estimate=jnp.array(0.0),
    beta=jnp.array(0.5),
    step=jnp.array(0),
  )
  
  # Test with small chunk size
  config_small_chunk = SMCConfig(
    template_sequence=sample_smc_config.template_sequence,
    population_size=sample_smc_config.population_size,
    n_states=sample_smc_config.n_states,
    generations=sample_smc_config.generations,
    mutation_rate=sample_smc_config.mutation_rate,
    diversification_ratio=sample_smc_config.diversification_ratio,
    sequence_type=sample_smc_config.sequence_type,
    annealing_schedule_config=sample_smc_config.annealing_schedule_config,
    fitness_evaluator=sample_smc_config.fitness_evaluator,
    memory_config=MemoryConfig(population_chunk_size=2),
  )
  
  next_state, metrics = smc_step(state, config_small_chunk)
  
  chex.assert_shape(next_state.population, state.population.shape)
  chex.assert_equal(jnp.isfinite(next_state.logZ_estimate), True)
  assert "mean_combined_fitness" in metrics


def test_chunked_processing_consistency():
  """Test that chunked processing gives consistent results."""
  key = random.PRNGKey(123)
  population = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=jnp.int8)
  
  fitness_evaluator = FitnessEvaluator(
    fitness_functions=(
      FitnessFunction(
        func=mock_fitness_fn,
        input_type="protein",
        name="test_fitness",
      ),
    ),
  )
  
  # Test with different chunk sizes
  fitness1, components1 = chunked_calculate_population_fitness(key, population, fitness_evaluator,  "protein", chunk_size=2)
  fitness2, components2 = chunked_calculate_population_fitness(key, population, fitness_evaluator,  "protein", chunk_size=4)

  # Results should be identical regardless of chunk size
  chex.assert_trees_all_close(fitness1, fitness2)
  chex.assert_trees_all_close(components1, components2)


def test_memory_config_integration():
  """Test that memory config settings are properly used."""
  key = random.PRNGKey(42)
  population = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int8)
  
  fitness_evaluator = FitnessEvaluator(
    fitness_functions=(
      FitnessFunction(
        func=mock_fitness_fn,
        input_type="protein",
        name="test_fitness",
      ),
    ),
  )
  
  # Create config with chunking disabled
  config_no_chunk = SMCConfig(
    template_sequence="MKY",
    population_size=10,
    n_states=20,
    generations=10,
    mutation_rate=0.1,
    diversification_ratio=0.1,
    sequence_type="protein",
    annealing_schedule_config=AnnealingScheduleConfig(linear_schedule, 1.0, 10),
    fitness_evaluator=fitness_evaluator,
    memory_config=MemoryConfig(enable_chunked_vmap=False),
  )
  
  state = SMCCarryState(
    key=key,
    population=population,
    logZ_estimate=jnp.array(0.0),
    beta=jnp.array(0.5),
  )
  
  # Should run without error regardless of chunking setting
  next_state, _ = smc_step(state, config_no_chunk)
  chex.assert_shape(next_state.population, state.population.shape)
