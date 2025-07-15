"""Tests for SMC data structures and PyTree functionality."""


import jax
import jax.numpy as jnp
import chex
import pytest
from jax import random

from proteinsmc.utils.data_structures import (
  MemoryConfig,
  SMCCarryState,
  SMCConfig,
  SMCOutput,
)
from proteinsmc.utils import AnnealingScheduleConfig, FitnessEvaluator, FitnessFunction, linear_schedule


def mock_fitness_fn(key, sequence, **kwargs) -> jnp.ndarray:
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
  
  memory_config = MemoryConfig(
    population_chunk_size=32,
    enable_chunked_vmap=True,
    device_memory_fraction=0.8,
  )
  
  return SMCConfig(
    seed_sequence="MKYN",
    population_size=100,
    n_states=20,
    generations=50,
    mutation_rate=0.1,
    diversification_ratio=0.2,
    sequence_type="protein",
    annealing_schedule=annealing_schedule_config,
    fitness_evaluator=sample_fitness_evaluator,
    memory_config=memory_config,
  )


def test_memory_config_pytree():
  """Test that MemoryConfig works as a PyTree."""
  config = MemoryConfig(
    population_chunk_size=64,
    enable_chunked_vmap=True,
    device_memory_fraction=0.9,
  )
  
  children, aux_data = config.tree_flatten()
  reconstructed = MemoryConfig.tree_unflatten(aux_data, children)
  
  chex.assert_equal(children, ())
  chex.assert_equal(reconstructed.population_chunk_size, config.population_chunk_size)
  chex.assert_equal(reconstructed.enable_chunked_vmap, config.enable_chunked_vmap)
  chex.assert_trees_all_close(reconstructed.device_memory_fraction, config.device_memory_fraction)


def test_smc_config_pytree(sample_smc_config):
  """Test that SMCConfig works as a PyTree."""
  # Test tree flattening and unflattening
  children, aux_data = sample_smc_config.tree_flatten()
  reconstructed = SMCConfig.tree_unflatten(aux_data, children)
  
  chex.assert_equal(children, ())  # Should be empty since all fields are auxiliary
  chex.assert_equal(reconstructed.seed_sequence, sample_smc_config.seed_sequence)
  chex.assert_equal(reconstructed.population_size, sample_smc_config.population_size)
  chex.assert_trees_all_close(reconstructed.mutation_rate, sample_smc_config.mutation_rate)


def test_smc_carry_state_pytree():
  """Test that SMCCarryState works as a PyTree."""
  key = random.PRNGKey(42)
  population = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32)
  
  state = SMCCarryState(
    key=key,
    population=population,
    logZ_estimate=jnp.array(1.5, dtype=jnp.float32),
    beta=jnp.array(0.8, dtype=jnp.float32),
    step=jnp.array(10, dtype=jnp.int32),
  )
  
  # Test tree flattening and unflattening
  children, aux_data = state.tree_flatten()
  reconstructed = SMCCarryState.tree_unflatten(aux_data, children)
  
  chex.assert_equal(len(children), 5)  # All fields are JAX arrays
  chex.assert_equal(aux_data, {})  # No auxiliary data
  chex.assert_trees_all_equal(reconstructed.population, population)
  chex.assert_trees_all_close(reconstructed.logZ_estimate, state.logZ_estimate)
  chex.assert_trees_all_close(reconstructed.beta, state.beta)


def test_smc_output_pytree(sample_smc_config):
  """Test that SMCOutput works as a PyTree."""
  generations = 10
  n_components = 2
  
  output = SMCOutput(
    input_config=sample_smc_config,
    mean_combined_fitness_per_gen=jnp.ones(generations),
    max_combined_fitness_per_gen=jnp.ones(generations) * 2,
    entropy_per_gen=jnp.ones(generations) * 0.5,
    beta_per_gen=jnp.linspace(0.1, 1.0, generations),
    ess_per_gen=jnp.ones(generations) * 0.8,
    fitness_components_per_gen=jnp.ones((generations, n_components)),
    final_logZhat=jnp.array(5.0),
    final_amino_acid_entropy=jnp.array(2.5),
  )
  
  # Test tree flattening and unflattening
  children, aux_data = output.tree_flatten()
  reconstructed = SMCOutput.tree_unflatten(aux_data, children)
  
  chex.assert_equal(len(children), 9)
  chex.assert_equal(children[0], sample_smc_config)
  chex.assert_trees_all_equal(reconstructed.mean_combined_fitness_per_gen, output.mean_combined_fitness_per_gen)
  chex.assert_trees_all_close(reconstructed.final_logZhat, output.final_logZhat)


def test_pytree_registration():
  """Test that all dataclasses are properly registered as PyTrees."""
  # Test that JAX can recognize our dataclasses as PyTrees
  key = random.PRNGKey(0)
  population = jnp.array([[1, 2, 3]], dtype=jnp.int32)
  
  state = SMCCarryState(
    key=key,
    population=population,
    logZ_estimate=jnp.array(0.0),
    beta=jnp.array(0.5),
  )
  
  # This should work without errors if PyTree registration is correct
  leaves = jax.tree_util.tree_leaves(state)
  chex.assert_equal(len(leaves), 5)  # key, population, logZ_estimate, beta, step
  
  # Test with JAX transformations
  def dummy_transform(state):
    return state.beta * 2

  # Stack the fields to create a batched PyTree for vmap
  states = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), state)
  result = jax.vmap(dummy_transform)(states)
  chex.assert_shape(result, (2,))


def test_memory_config_defaults():
  """Test MemoryConfig default values."""
  config = MemoryConfig()
  
  chex.assert_equal(config.population_chunk_size, 64)
  chex.assert_equal(config.enable_chunked_vmap, True)
  chex.assert_trees_all_close(config.device_memory_fraction, 0.8)


def test_smc_config_validation_ready(sample_smc_config):
  """Test that SMCConfig contains all necessary fields for validation."""
  # Ensure all required fields are present
  chex.assert_equal(hasattr(sample_smc_config, 'seed_sequence'), True)
  chex.assert_equal(hasattr(sample_smc_config, 'population_size'), True)
  chex.assert_equal(hasattr(sample_smc_config, 'mutation_rate'), True)
  chex.assert_equal(hasattr(sample_smc_config, 'sequence_type'), True)
  chex.assert_equal(hasattr(sample_smc_config, 'fitness_evaluator'), True)
  chex.assert_equal(hasattr(sample_smc_config, 'memory_config'), True)
  
  # Test that values are reasonable
  chex.assert_equal(sample_smc_config.population_size > 0, True)
  chex.assert_equal(0 <= sample_smc_config.mutation_rate <= 1, True)
  assert sample_smc_config.sequence_type in ["protein", "nucleotide"]


def test_smc_carry_state_default_step():
  """Test that SMCCarryState has correct default step value."""
  key = random.PRNGKey(0)
  population = jnp.array([[1, 2, 3]], dtype=jnp.int32)
  
  state = SMCCarryState(
    key=key,
    population=population,
    logZ_estimate=jnp.array(0.0),
    beta=jnp.array(0.5),
  )
  
  chex.assert_equal(state.step, 0)
  assert isinstance(state.step, jax.Array)
  chex.assert_equal(state.step.dtype, jnp.int32)
