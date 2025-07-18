"""Tests for SMC model classes."""

from __future__ import annotations

from unittest.mock import Mock

import chex
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Float

from proteinsmc.models.annealing import AnnealingConfig
from proteinsmc.models.fitness import FitnessEvaluator, FitnessFunction
from proteinsmc.models.memory import MemoryConfig
from proteinsmc.models.smc import SMCState, SMCConfig, SMCOutput


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
def mock_annealing_schedule() -> AnnealingConfig:
  """Create a mock AnnealingScheduleConfig for testing."""
  return AnnealingConfig(
    annealing_fn="linear",
    beta_max=1.0,
    n_steps=10,
  )


@pytest.fixture
def valid_smc_config(
  mock_fitness_evaluator: FitnessEvaluator,
  mock_memory_config: MemoryConfig,
  mock_annealing_schedule: AnnealingConfig,
) -> SMCConfig:
  """Create a valid SMCConfig for testing."""
  return SMCConfig(
    seed_sequence="MKLLVL",
    num_samples=10,
    n_states=100,
    mutation_rate=0.1,
    diversification_ratio=0.2,
    sequence_type="protein",
    fitness_evaluator=mock_fitness_evaluator,
    memory_config=mock_memory_config,
    population_size=50,
    annealing_schedule=mock_annealing_schedule,
  )


class TestSMCConfig:
  """Test cases for SMCConfig."""

  def test_init_success(self, valid_smc_config: SMCConfig) -> None:
    """Test successful SMCConfig initialization."""
    config = valid_smc_config
    assert config.seed_sequence == "MKLLVL"
    assert config.num_samples == 10
    assert config.n_states == 100
    assert config.mutation_rate == 0.1
    assert config.diversification_ratio == 0.2
    assert config.sequence_type == "protein"
    assert config.population_size == 50

  def test_validation_negative_population_size(
    self,
    mock_fitness_evaluator: FitnessEvaluator,
    mock_memory_config: MemoryConfig,
    mock_annealing_schedule: AnnealingConfig,
  ) -> None:
    """Test validation fails for negative population_size."""
    with pytest.raises(ValueError, match="population_size must be positive"):
      SMCConfig(
        seed_sequence="MKLLVL",
        num_samples=10,
        n_states=100,
        mutation_rate=0.1,
        diversification_ratio=0.2,
        sequence_type="protein",
        fitness_evaluator=mock_fitness_evaluator,
        memory_config=mock_memory_config,
        population_size=-1,
        annealing_schedule=mock_annealing_schedule,
      )

  def test_validation_zero_population_size(
    self,
    mock_fitness_evaluator: FitnessEvaluator,
    mock_memory_config: MemoryConfig,
    mock_annealing_schedule: AnnealingConfig,
  ) -> None:
    """Test validation fails for zero population_size."""
    with pytest.raises(ValueError, match="population_size must be positive"):
      SMCConfig(
        seed_sequence="MKLLVL",
        num_samples=10,
        n_states=100,
        mutation_rate=0.1,
        diversification_ratio=0.2,
        sequence_type="protein",
        fitness_evaluator=mock_fitness_evaluator,
        memory_config=mock_memory_config,
        population_size=0,
        annealing_schedule=mock_annealing_schedule,
      )

  def test_type_validation_population_size(
    self,
    mock_fitness_evaluator: FitnessEvaluator,
    mock_memory_config: MemoryConfig,
    mock_annealing_schedule: AnnealingConfig,
  ) -> None:
    """Test type validation for population_size."""
    with pytest.raises(TypeError, match="population_size must be an integer"):
      SMCConfig(
        seed_sequence="MKLLVL",
        num_samples=10,
        n_states=100,
        mutation_rate=0.1,
        diversification_ratio=0.2,
        sequence_type="protein",
        fitness_evaluator=mock_fitness_evaluator,
        memory_config=mock_memory_config,
        population_size=50.5,  # type: ignore
        annealing_schedule=mock_annealing_schedule,
      )

  def test_type_validation_annealing_schedule(
    self,
    mock_fitness_evaluator: FitnessEvaluator,
    mock_memory_config: MemoryConfig,
  ) -> None:
    """Test type validation for annealing_schedule."""
    with pytest.raises(TypeError, match="annealing_schedule must be an AnnealingScheduleConfig"):
      SMCConfig(
        seed_sequence="MKLLVL",
        num_samples=10,
        n_states=100,
        mutation_rate=0.1,
        diversification_ratio=0.2,
        sequence_type="protein",
        fitness_evaluator=mock_fitness_evaluator,
        memory_config=mock_memory_config,
        population_size=50,
        annealing_schedule="not_config",  # type: ignore
      )

  def test_additional_config_fields(self, valid_smc_config: SMCConfig) -> None:
    """Test the additional_config_fields property."""
    config = valid_smc_config
    fields = config.additional_config_fields
    assert "population_size" in fields
    assert "annealing_schedule" in fields
    assert fields["population_size"] == "population_size"
    assert fields["annealing_schedule"] == "annealing_schedule"

  def test_pytree_registration(self, valid_smc_config: SMCConfig) -> None:
    """Test that SMCConfig is properly registered as a PyTree."""
    config = valid_smc_config

    # Test that it can be used in JAX transformations
    def process_config(c: SMCConfig) -> int:
      return c.population_size

    jitted_process = jax.jit(process_config)
    result = jitted_process(config)
    chex.assert_trees_all_close(result, 50)

    # Test tree flatten/unflatten
    leaves, treedef = jax.tree_util.tree_flatten(config)
    unflattened_config = jax.tree_util.tree_unflatten(treedef, leaves)

    assert config.seed_sequence == unflattened_config.seed_sequence
    assert config.num_samples == unflattened_config.generations
    assert config.population_size == unflattened_config.population_size
    assert config.annealing_schedule.annealing_fn == unflattened_config.annealing_schedule.schedule_fn


class TestSMCCarryState:
  """Test cases for SMCCarryState."""

  @pytest.fixture
  def sample_key(self) -> jax.Array:
    """Create a sample PRNG key."""
    return jax.random.PRNGKey(42)

  @pytest.fixture
  def sample_population(self) -> jax.Array:
    """Create a sample population array."""
    return jnp.array([[1, 2, 3], [4, 5, 6]])

  @pytest.fixture
  def sample_carry_state(
    self,
    sample_key: jax.Array,
    sample_population: jax.Array,
  ) -> SMCState:
    """Create a sample SMCCarryState."""
    return SMCState(
      key=sample_key,
      population=sample_population,
      logZ_estimate=jnp.array(1.5, dtype=jnp.float32),
      beta=jnp.array(0.8, dtype=jnp.float32),
      step=jnp.array(5, dtype=jnp.int32),
    )

  def test_init_success(self, sample_carry_state: SMCState) -> None:
    """Test successful SMCCarryState initialization."""
    state = sample_carry_state
    assert state.key is not None
    assert state.population is not None
    chex.assert_trees_all_close(state.logZ_estimate, 1.5)
    chex.assert_trees_all_close(state.beta, 0.8)
    chex.assert_trees_all_close(state.step, 5)

  def test_init_with_default_step(
    self,
    sample_key: jax.Array,
    sample_population: jax.Array,
  ) -> None:
    """Test SMCCarryState initialization with default step."""
    state = SMCState(
      key=sample_key,
      population=sample_population,
      logZ_estimate=jnp.array(1.0, dtype=jnp.float32),
      beta=jnp.array(0.5, dtype=jnp.float32),
    )
    chex.assert_trees_all_close(state.step, 0)

  def test_pytree_registration(self, sample_carry_state: SMCState) -> None:
    """Test that SMCCarryState is properly registered as a PyTree."""
    state = sample_carry_state

    # Test that it can be used in JAX transformations
    def process_state(s: SMCState) -> Float:
      return s.beta

    jitted_process = jax.jit(process_state)
    result = jitted_process(state)
    chex.assert_trees_all_close(result, 0.8)

    # Test tree flatten/unflatten
    leaves, treedef = jax.tree_util.tree_flatten(state)
    unflattened_state = jax.tree_util.tree_unflatten(treedef, leaves)

    # Check that key arrays are equal (need special handling for PRNG keys)
    chex.assert_trees_all_close(state.key, unflattened_state.key)
    chex.assert_trees_all_close(state.population, unflattened_state.population)
    chex.assert_trees_all_close(state.logZ_estimate, unflattened_state.logZ_estimate)
    chex.assert_trees_all_close(state.beta, unflattened_state.beta)
    chex.assert_trees_all_close(state.step, unflattened_state.step)


class TestSMCOutput:
  """Test cases for SMCOutput."""

  @pytest.fixture
  def sample_output(self, valid_smc_config: SMCConfig) -> SMCOutput:
    """Create a sample SMCOutput."""
    return SMCOutput(
      input_config=valid_smc_config,
      mean_combined_fitness_per_gen=jnp.array([1.0, 1.1, 1.2], dtype=jnp.float32),
      max_combined_fitness_per_gen=jnp.array([1.5, 1.6, 1.7], dtype=jnp.float32),
      entropy_per_gen=jnp.array([2.0, 2.1, 2.2], dtype=jnp.float32),
      beta_per_gen=jnp.array([0.1, 0.5, 1.0], dtype=jnp.float32),
      ess_per_gen=jnp.array([45.0, 46.0, 47.0], dtype=jnp.float32),
      fitness_components_per_gen=jnp.array([[1.0, 1.1], [1.2, 1.3], [1.4, 1.5]], dtype=jnp.float32),
      final_logZhat=jnp.array(3.5, dtype=jnp.float32),
      final_amino_acid_entropy=jnp.array(2.8, dtype=jnp.float32),
    )

  def test_init_success(self, sample_output: SMCOutput) -> None:
    """Test successful SMCOutput initialization."""
    output = sample_output
    assert output.input_config is not None
    chex.assert_trees_all_close(output.final_logZhat, 3.5)
    chex.assert_trees_all_close(output.final_amino_acid_entropy, 2.8)

  def test_input_configs_property(
    self,
    sample_output: SMCOutput,
    valid_smc_config: SMCConfig,
  ) -> None:
    """Test the input_configs property."""
    configs = sample_output.input_configs
    assert configs == valid_smc_config

  def test_per_gen_stats_metrics_property(self, sample_output: SMCOutput) -> None:
    """Test the per_gen_stats_metrics property."""
    metrics = sample_output.per_gen_stats_metrics
    expected_metrics = {
      "mean_fitness": "mean_combined_fitness_per_gen",
      "max_fitness": "max_combined_fitness_per_gen",
      "entropy": "entropy_per_gen",
      "beta": "beta_per_gen",
      "ess": "ess_per_gen",
      "fitness_components": "fitness_components_per_gen",
    }
    assert metrics == expected_metrics

  def test_summary_stats_metrics_property(self, sample_output: SMCOutput) -> None:
    """Test the summary_stats_metrics property."""
    metrics = sample_output.summary_stats_metrics
    expected_metrics = {
      "final_logZhat": "final_logZhat",
      "final_amino_acid_entropy": "final_amino_acid_entropy",
    }
    assert metrics == expected_metrics

  def test_output_type_name_property(self, sample_output: SMCOutput) -> None:
    """Test the output_type_name property."""
    assert sample_output.output_type_name == "SMC"

  def test_pytree_registration(self, sample_output: SMCOutput) -> None:
    """Test that SMCOutput is properly registered as a PyTree."""
    output = sample_output

    # Test that it can be used in JAX transformations
    def process_output(o: SMCOutput) -> Float:
      return o.final_logZhat

    jitted_process = jax.jit(process_output)
    result = jitted_process(output)
    chex.assert_trees_all_close(result, 3.5)

    # Test tree flatten/unflatten
    leaves, treedef = jax.tree_util.tree_flatten(output)
    unflattened_output = jax.tree_util.tree_unflatten(treedef, leaves)

    chex.assert_trees_all_close(output.final_logZhat, unflattened_output.final_logZhat)
    chex.assert_trees_all_close(
      output.final_amino_acid_entropy,
      unflattened_output.final_amino_acid_entropy,
    )
    chex.assert_trees_all_close(
      output.mean_combined_fitness_per_gen,
      unflattened_output.mean_combined_fitness_per_gen,
    )
