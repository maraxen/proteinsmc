"""Tests for SMC data structures and PyTree functionality."""

import jax
import jax.numpy as jnp
import pytest
from jax import random

from proteinsmc.sampling.smc.data_structures import (
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
    template_sequence="MKYN",
    population_size=100,
    n_states=20,
    generations=50,
    mutation_rate=0.1,
    diversification_ratio=0.2,
    sequence_type="protein",
    annealing_schedule_config=annealing_schedule_config,
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
  
  assert children == ()  # Should be empty since all fields are auxiliary
  assert reconstructed.population_chunk_size == config.population_chunk_size
  assert reconstructed.enable_chunked_vmap == config.enable_chunked_vmap
  assert reconstructed.device_memory_fraction == config.device_memory_fraction


def test_smc_config_pytree(sample_smc_config):
  """Test that SMCConfig works as a PyTree."""
  # Test tree flattening and unflattening
  children, aux_data = sample_smc_config.tree_flatten()
  reconstructed = SMCConfig.tree_unflatten(aux_data, children)
  
  assert children == ()  # Should be empty since all fields are auxiliary
  assert reconstructed.template_sequence == sample_smc_config.template_sequence
  assert reconstructed.population_size == sample_smc_config.population_size
  assert reconstructed.mutation_rate == sample_smc_config.mutation_rate


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
  
  assert len(children) == 5  # All fields are JAX arrays
  assert aux_data == {}  # No auxiliary data
  assert jnp.array_equal(reconstructed.population, population)
  assert reconstructed.logZ_estimate == state.logZ_estimate
  assert reconstructed.beta == state.beta


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
  
  assert len(children) == 7  # Number of JAX array fields
  assert "final_logZhat" in aux_data
  assert "final_amino_acid_entropy" in aux_data
  assert jnp.array_equal(reconstructed.mean_combined_fitness_per_gen, output.mean_combined_fitness_per_gen)
  assert reconstructed.final_logZhat == output.final_logZhat


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
  assert len(leaves) == 5  # key, population, logZ_estimate, beta, step
  
  # Test with JAX transformations
  def dummy_transform(state):
    return state.beta * 2

  # Stack the fields to create a batched PyTree for vmap
  states = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), state)
  result = jax.vmap(dummy_transform)(states)
  assert result.shape == (2,)


def test_memory_config_defaults():
  """Test MemoryConfig default values."""
  config = MemoryConfig()
  
  assert config.population_chunk_size == 64
  assert config.enable_chunked_vmap is True
  assert config.device_memory_fraction == 0.8


def test_smc_config_validation_ready(sample_smc_config):
  """Test that SMCConfig contains all necessary fields for validation."""
  # Ensure all required fields are present
  assert hasattr(sample_smc_config, 'template_sequence')
  assert hasattr(sample_smc_config, 'population_size')
  assert hasattr(sample_smc_config, 'mutation_rate')
  assert hasattr(sample_smc_config, 'sequence_type')
  assert hasattr(sample_smc_config, 'fitness_evaluator')
  assert hasattr(sample_smc_config, 'memory_config')
  
  # Test that values are reasonable
  assert sample_smc_config.population_size > 0
  assert 0 <= sample_smc_config.mutation_rate <= 1
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
  
  assert state.step == 0
  assert isinstance(state.step, jax.Array)
  assert state.step.dtype == jnp.int32
