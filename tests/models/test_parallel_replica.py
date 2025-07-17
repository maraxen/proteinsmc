"""Tests for parallel replica exchange model classes."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Float

from proteinsmc.models.annealing import AnnealingConfig
from proteinsmc.models.fitness import FitnessEvaluator, FitnessFunction
from proteinsmc.models.memory import MemoryConfig
from proteinsmc.models.parallel_replica import (
  ExchangeConfig,
  IslandState,
  PRSMCCarryState,
  PRSMCStepConfig,
  ParallelReplicaConfig,
  ParallelReplicaSMCOutput,
)


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
def valid_exchange_config(mock_fitness_evaluator: FitnessEvaluator) -> ExchangeConfig:
  """Create a valid ExchangeConfig for testing."""
  return ExchangeConfig(
    population_size_per_island=25,
    n_islands=4,
    n_exchange_attempts=10,
    fitness_evaluator=mock_fitness_evaluator,
    exchange_frequency=0.1,
    sequence_type="protein",
    n_exchange_attempts_per_cycle=2,
    ess_threshold_fraction=0.8,
  )


@pytest.fixture
def valid_prsmc_step_config(
  mock_fitness_evaluator: FitnessEvaluator,
  mock_annealing_schedule: AnnealingConfig,
  valid_exchange_config: ExchangeConfig,
) -> PRSMCStepConfig:
  """Create a valid PRSMCStepConfig for testing."""
  return PRSMCStepConfig(
    population_size_per_island=25,
    mutation_rate=0.1,
    fitness_evaluator=mock_fitness_evaluator,
    sequence_type="protein",
    ess_threshold_frac=0.8,
    meta_beta_annealing_schedule=mock_annealing_schedule,
    exchange_config=valid_exchange_config,
  )


@pytest.fixture
def valid_parallel_replica_config(
  mock_fitness_evaluator: FitnessEvaluator,
  mock_memory_config: MemoryConfig,
  valid_prsmc_step_config: PRSMCStepConfig,
) -> ParallelReplicaConfig:
  """Create a valid ParallelReplicaConfig for testing."""
  return ParallelReplicaConfig(
    seed_sequence="MKLLVL",
    generations=10,
    n_states=100,
    mutation_rate=0.1,
    diversification_ratio=0.2,
    sequence_type="protein",
    fitness_evaluator=mock_fitness_evaluator,
    memory_config=mock_memory_config,
    population_size_per_island=25,
    n_islands=4,
    island_betas=[0.0, 0.3, 0.7, 1.0],
    step_config=valid_prsmc_step_config,
  )


class TestExchangeConfig:
  """Test cases for ExchangeConfig."""

  def test_init_success(self, valid_exchange_config: ExchangeConfig) -> None:
    """Test successful ExchangeConfig initialization."""
    config = valid_exchange_config
    assert config.population_size_per_island == 25
    assert config.n_islands == 4
    assert config.n_exchange_attempts == 10
    assert config.exchange_frequency == 0.1
    assert config.sequence_type == "protein"
    assert config.n_exchange_attempts_per_cycle == 2
    assert config.ess_threshold_fraction == 0.8

  def test_validation_negative_population_size_per_island(
    self,
    mock_fitness_evaluator: FitnessEvaluator,
  ) -> None:
    """Test validation fails for negative population_size_per_island."""
    with pytest.raises(ValueError, match="population_size_per_island must be positive"):
      ExchangeConfig(
        population_size_per_island=-1,
        n_islands=4,
        n_exchange_attempts=10,
        fitness_evaluator=mock_fitness_evaluator,
        exchange_frequency=0.1,
        sequence_type="protein",
        n_exchange_attempts_per_cycle=2,
        ess_threshold_fraction=0.8,
      )

  def test_validation_negative_n_islands(self, mock_fitness_evaluator: FitnessEvaluator) -> None:
    """Test validation fails for negative n_islands."""
    with pytest.raises(ValueError, match="n_islands must be positive"):
      ExchangeConfig(
        population_size_per_island=25,
        n_islands=-1,
        n_exchange_attempts=10,
        fitness_evaluator=mock_fitness_evaluator,
        exchange_frequency=0.1,
        sequence_type="protein",
        n_exchange_attempts_per_cycle=2,
        ess_threshold_fraction=0.8,
      )

  def test_validation_invalid_exchange_frequency_high(
    self,
    mock_fitness_evaluator: FitnessEvaluator,
  ) -> None:
    """Test validation fails for exchange_frequency > 1.0."""
    with pytest.raises(ValueError, match="exchange_frequency must be in \\[0.0, 1.0\\]"):
      ExchangeConfig(
        population_size_per_island=25,
        n_islands=4,
        n_exchange_attempts=10,
        fitness_evaluator=mock_fitness_evaluator,
        exchange_frequency=1.5,
        sequence_type="protein",
        n_exchange_attempts_per_cycle=2,
        ess_threshold_fraction=0.8,
      )

  def test_validation_invalid_exchange_frequency_low(
    self,
    mock_fitness_evaluator: FitnessEvaluator,
  ) -> None:
    """Test validation fails for exchange_frequency < 0.0."""
    with pytest.raises(ValueError, match="exchange_frequency must be in \\[0.0, 1.0\\]"):
      ExchangeConfig(
        population_size_per_island=25,
        n_islands=4,
        n_exchange_attempts=10,
        fitness_evaluator=mock_fitness_evaluator,
        exchange_frequency=-0.1,
        sequence_type="protein",
        n_exchange_attempts_per_cycle=2,
        ess_threshold_fraction=0.8,
      )

  def test_validation_invalid_sequence_type(
    self,
    mock_fitness_evaluator: FitnessEvaluator,
  ) -> None:
    """Test validation fails for invalid sequence_type."""
    with pytest.raises(ValueError, match="sequence_type must be 'protein' or 'nucleotide'"):
      ExchangeConfig(
        population_size_per_island=25,
        n_islands=4,
        n_exchange_attempts=10,
        fitness_evaluator=mock_fitness_evaluator,
        exchange_frequency=0.1,
        sequence_type="invalid",  # type: ignore
        n_exchange_attempts_per_cycle=2,
        ess_threshold_fraction=0.8,
      )

  def test_type_validation_population_size_per_island(
    self,
    mock_fitness_evaluator: FitnessEvaluator,
  ) -> None:
    """Test type validation for population_size_per_island."""
    with pytest.raises(TypeError, match="population_size_per_island must be an integer"):
      ExchangeConfig(
        population_size_per_island=25.5,  # type: ignore
        n_islands=4,
        n_exchange_attempts=10,
        fitness_evaluator=mock_fitness_evaluator,
        exchange_frequency=0.1,
        sequence_type="protein",
        n_exchange_attempts_per_cycle=2,
        ess_threshold_fraction=0.8,
      )

  def test_pytree_registration(self, valid_exchange_config: ExchangeConfig) -> None:
    """Test that ExchangeConfig is properly registered as a PyTree."""
    config = valid_exchange_config

    # Test that it can be used in JAX transformations
    def process_config(c: ExchangeConfig) -> int:
      return c.n_islands

    jitted_process = jax.jit(process_config)
    result = jitted_process(config)
    chex.assert_trees_all_close(result, 4)

    # Test tree flatten/unflatten
    leaves, treedef = jax.tree_util.tree_flatten(config)
    unflattened_config = jax.tree_util.tree_unflatten(treedef, leaves)

    assert config.population_size_per_island == unflattened_config.population_size_per_island
    assert config.n_islands == unflattened_config.n_islands
    assert config.n_exchange_attempts == unflattened_config.n_exchange_attempts
    assert config.sequence_type == unflattened_config.sequence_type


class TestPRSMCStepConfig:
  """Test cases for PRSMCStepConfig."""

  def test_init_success(self, valid_prsmc_step_config: PRSMCStepConfig) -> None:
    """Test successful PRSMCStepConfig initialization."""
    config = valid_prsmc_step_config
    assert config.population_size_per_island == 25
    assert config.mutation_rate == 0.1
    assert config.sequence_type == "protein"
    assert config.ess_threshold_frac == 0.8

  def test_validation_negative_population_size_per_island(
    self,
    mock_fitness_evaluator: FitnessEvaluator,
    mock_annealing_schedule: AnnealingConfig,
    valid_exchange_config: ExchangeConfig,
  ) -> None:
    """Test validation fails for negative population_size_per_island."""
    with pytest.raises(ValueError, match="population_size_per_island must be positive"):
      PRSMCStepConfig(
        population_size_per_island=-1,
        mutation_rate=0.1,
        fitness_evaluator=mock_fitness_evaluator,
        sequence_type="protein",
        ess_threshold_frac=0.8,
        meta_beta_annealing_schedule=mock_annealing_schedule,
        exchange_config=valid_exchange_config,
      )

  def test_validation_invalid_mutation_rate_high(
    self,
    mock_fitness_evaluator: FitnessEvaluator,
    mock_annealing_schedule: AnnealingConfig,
    valid_exchange_config: ExchangeConfig,
  ) -> None:
    """Test validation fails for mutation_rate > 1.0."""
    with pytest.raises(ValueError, match="mutation_rate must be in \\[0.0, 1.0\\]"):
      PRSMCStepConfig(
        population_size_per_island=25,
        mutation_rate=1.5,
        fitness_evaluator=mock_fitness_evaluator,
        sequence_type="protein",
        ess_threshold_frac=0.8,
        meta_beta_annealing_schedule=mock_annealing_schedule,
        exchange_config=valid_exchange_config,
      )

  def test_validation_invalid_ess_threshold_frac_high(
    self,
    mock_fitness_evaluator: FitnessEvaluator,
    mock_annealing_schedule: AnnealingConfig,
    valid_exchange_config: ExchangeConfig,
  ) -> None:
    """Test validation fails for ess_threshold_frac > 1.0."""
    with pytest.raises(ValueError, match="ess_threshold_frac must be in \\(0.0, 1.0\\]"):
      PRSMCStepConfig(
        population_size_per_island=25,
        mutation_rate=0.1,
        fitness_evaluator=mock_fitness_evaluator,
        sequence_type="protein",
        ess_threshold_frac=1.5,
        meta_beta_annealing_schedule=mock_annealing_schedule,
        exchange_config=valid_exchange_config,
      )

  def test_validation_invalid_ess_threshold_frac_low(
    self,
    mock_fitness_evaluator: FitnessEvaluator,
    mock_annealing_schedule: AnnealingConfig,
    valid_exchange_config: ExchangeConfig,
  ) -> None:
    """Test validation fails for ess_threshold_frac <= 0.0."""
    with pytest.raises(ValueError, match="ess_threshold_frac must be in \\(0.0, 1.0\\]"):
      PRSMCStepConfig(
        population_size_per_island=25,
        mutation_rate=0.1,
        fitness_evaluator=mock_fitness_evaluator,
        sequence_type="protein",
        ess_threshold_frac=0.0,
        meta_beta_annealing_schedule=mock_annealing_schedule,
        exchange_config=valid_exchange_config,
      )

  def test_pytree_registration(self, valid_prsmc_step_config: PRSMCStepConfig) -> None:
    """Test that PRSMCStepConfig is properly registered as a PyTree."""
    config = valid_prsmc_step_config

    # Test that it can be used in JAX transformations
    def process_config(c: PRSMCStepConfig) -> int:
      return c.population_size_per_island

    jitted_process = jax.jit(process_config)
    result = jitted_process(config)
    chex.assert_trees_all_close(result, 25)

    # Test tree flatten/unflatten
    leaves, treedef = jax.tree_util.tree_flatten(config)
    unflattened_config = jax.tree_util.tree_unflatten(treedef, leaves)

    assert config.population_size_per_island == unflattened_config.population_size_per_island
    assert config.mutation_rate == unflattened_config.mutation_rate
    assert config.sequence_type == unflattened_config.sequence_type
    assert config.ess_threshold_frac == unflattened_config.ess_threshold_frac


class TestParallelReplicaConfig:
  """Test cases for ParallelReplicaConfig."""

  def test_init_success(self, valid_parallel_replica_config: ParallelReplicaConfig) -> None:
    """Test successful ParallelReplicaConfig initialization."""
    config = valid_parallel_replica_config
    assert config.seed_sequence == "MKLLVL"
    assert config.generations == 10
    assert config.n_islands == 4
    assert config.island_betas == [0.0, 0.3, 0.7, 1.0]
    assert config.population_size_per_island == 25

  def test_validation_negative_n_islands(
    self,
    mock_fitness_evaluator: FitnessEvaluator,
    mock_memory_config: MemoryConfig,
    valid_prsmc_step_config: PRSMCStepConfig,
  ) -> None:
    """Test validation fails for negative n_islands."""
    with pytest.raises(ValueError, match="n_islands must be positive"):
      ParallelReplicaConfig(
        seed_sequence="MKLLVL",
        generations=10,
        n_states=100,
        mutation_rate=0.1,
        diversification_ratio=0.2,
        sequence_type="protein",
        fitness_evaluator=mock_fitness_evaluator,
        memory_config=mock_memory_config,
        population_size_per_island=25,
        n_islands=-1,
        island_betas=[0.0, 0.5, 1.0],
        step_config=valid_prsmc_step_config,
      )

  def test_validation_mismatched_island_betas_length(
    self,
    mock_fitness_evaluator: FitnessEvaluator,
    mock_memory_config: MemoryConfig,
    valid_prsmc_step_config: PRSMCStepConfig,
  ) -> None:
    """Test validation fails when island_betas length doesn't match n_islands."""
    with pytest.raises(ValueError, match="Length of island_betas \\(3\\) must match n_islands \\(4\\)"):
      ParallelReplicaConfig(
        seed_sequence="MKLLVL",
        generations=10,
        n_states=100,
        mutation_rate=0.1,
        diversification_ratio=0.2,
        sequence_type="protein",
        fitness_evaluator=mock_fitness_evaluator,
        memory_config=mock_memory_config,
        population_size_per_island=25,
        n_islands=4,
        island_betas=[0.0, 0.5, 1.0],  # Only 3 values for 4 islands
        step_config=valid_prsmc_step_config,
      )

  def test_validation_invalid_island_betas_values(
    self,
    mock_fitness_evaluator: FitnessEvaluator,
    mock_memory_config: MemoryConfig,
    valid_prsmc_step_config: PRSMCStepConfig,
  ) -> None:
    """Test validation fails for island_betas outside [0.0, 1.0]."""
    with pytest.raises(ValueError, match="All island_betas must be in \\[0.0, 1.0\\]"):
      ParallelReplicaConfig(
        seed_sequence="MKLLVL",
        generations=10,
        n_states=100,
        mutation_rate=0.1,
        diversification_ratio=0.2,
        sequence_type="protein",
        fitness_evaluator=mock_fitness_evaluator,
        memory_config=mock_memory_config,
        population_size_per_island=25,
        n_islands=4,
        island_betas=[0.0, 0.5, 1.0, 1.5],  # 1.5 is invalid
        step_config=valid_prsmc_step_config,
      )

  def test_type_validation_island_betas_not_list(
    self,
    mock_fitness_evaluator: FitnessEvaluator,
    mock_memory_config: MemoryConfig,
    valid_prsmc_step_config: PRSMCStepConfig,
  ) -> None:
    """Test type validation for island_betas not being a list."""
    with pytest.raises(TypeError, match="island_betas must be a list of floats"):
      ParallelReplicaConfig(
        seed_sequence="MKLLVL",
        generations=10,
        n_states=100,
        mutation_rate=0.1,
        diversification_ratio=0.2,
        sequence_type="protein",
        fitness_evaluator=mock_fitness_evaluator,
        memory_config=mock_memory_config,
        population_size_per_island=25,
        n_islands=4,
        island_betas=(0.0, 0.3, 0.7, 1.0),  # type: ignore  # Tuple instead of list
        step_config=valid_prsmc_step_config,
      )

  def test_type_validation_island_betas_content(
    self,
    mock_fitness_evaluator: FitnessEvaluator,
    mock_memory_config: MemoryConfig,
    valid_prsmc_step_config: PRSMCStepConfig,
  ) -> None:
    """Test type validation for island_betas containing non-floats."""
    with pytest.raises(TypeError, match="island_betas must be a list of floats"):
      ParallelReplicaConfig(
        seed_sequence="MKLLVL",
        generations=10,
        n_states=100,
        mutation_rate=0.1,
        diversification_ratio=0.2,
        sequence_type="protein",
        fitness_evaluator=mock_fitness_evaluator,
        memory_config=mock_memory_config,
        population_size_per_island=25,
        n_islands=4,
        island_betas=[0.0, "0.3", 0.7, 1.0],  # type: ignore  # String instead of float
        step_config=valid_prsmc_step_config,
      )

  def test_additional_config_fields(
    self,
    valid_parallel_replica_config: ParallelReplicaConfig,
  ) -> None:
    """Test the additional_config_fields property."""
    config = valid_parallel_replica_config
    fields = config.additional_config_fields
    assert "n_islands" in fields
    assert "island_betas" in fields
    assert "island_config" in fields

  def test_pytree_registration(self, valid_parallel_replica_config: ParallelReplicaConfig) -> None:
    """Test that ParallelReplicaConfig is properly registered as a PyTree."""
    config = valid_parallel_replica_config

    # Test that it can be used in JAX transformations
    def process_config(c: ParallelReplicaConfig) -> int:
      return c.n_islands

    jitted_process = jax.jit(process_config)
    result = jitted_process(config)
    chex.assert_trees_all_close(result, 4)

    # Test tree flatten/unflatten
    leaves, treedef = jax.tree_util.tree_flatten(config)
    unflattened_config = jax.tree_util.tree_unflatten(treedef, leaves)

    assert config.seed_sequence == unflattened_config.seed_sequence
    assert config.generations == unflattened_config.generations
    assert config.n_islands == unflattened_config.n_islands
    # Note: island_betas is converted from tuple back to list during unflatten
    assert list(config.island_betas) == list(unflattened_config.island_betas)


class TestIslandState:
  """Test cases for IslandState."""

  @pytest.fixture
  def sample_key(self) -> jax.Array:
    """Create a sample PRNG key."""
    return jax.random.PRNGKey(42)

  @pytest.fixture
  def sample_island_state(self, sample_key: jax.Array) -> IslandState:
    """Create a sample IslandState."""
    return IslandState(
      key=sample_key,
      population=jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32),
      beta=jnp.array(0.8, dtype=jnp.float32),
      logZ_estimate=jnp.array(1.5, dtype=jnp.float32),
      ess=jnp.array(45.0, dtype=jnp.float32),
      mean_fitness=jnp.array(2.5, dtype=jnp.float32),
      step=jnp.array(5, dtype=jnp.int32),
    )

  def test_init_success(self, sample_island_state: IslandState) -> None:
    """Test successful IslandState initialization."""
    state = sample_island_state
    assert state.key is not None
    chex.assert_trees_all_close(state.beta, 0.8)
    chex.assert_trees_all_close(state.logZ_estimate, 1.5)
    chex.assert_trees_all_close(state.ess, 45.0)
    chex.assert_trees_all_close(state.mean_fitness, 2.5)
    chex.assert_trees_all_close(state.step, 5)

  def test_pytree_registration(self, sample_island_state: IslandState) -> None:
    """Test that IslandState is properly registered as a PyTree."""
    state = sample_island_state

    # Test that it can be used in JAX transformations
    def process_state(s: IslandState) -> Float:
      return s.beta

    jitted_process = jax.jit(process_state)
    result = jitted_process(state)
    chex.assert_trees_all_close(result, 0.8)

    # Test tree flatten/unflatten
    leaves, treedef = jax.tree_util.tree_flatten(state)
    unflattened_state = jax.tree_util.tree_unflatten(treedef, leaves)

    chex.assert_trees_all_close(state.key, unflattened_state.key)
    chex.assert_trees_all_close(state.population, unflattened_state.population)
    chex.assert_trees_all_close(state.beta, unflattened_state.beta)
    chex.assert_trees_all_close(state.logZ_estimate, unflattened_state.logZ_estimate)
    chex.assert_trees_all_close(state.ess, unflattened_state.ess)
    chex.assert_trees_all_close(state.mean_fitness, unflattened_state.mean_fitness)
    chex.assert_trees_all_close(state.step, unflattened_state.step)


class TestPRSMCCarryState:
  """Test cases for PRSMCCarryState."""

  @pytest.fixture
  def sample_carry_state(self) -> PRSMCCarryState:
    """Create a sample PRSMCCarryState."""
    key = jax.random.PRNGKey(42)
    overall_state = IslandState(
      key=key,
      population=jnp.array([[1, 2], [3, 4]], dtype=jnp.int32),
      beta=jnp.array(0.5, dtype=jnp.float32),
      logZ_estimate=jnp.array(1.0, dtype=jnp.float32),
      ess=jnp.array(40.0, dtype=jnp.float32),
      mean_fitness=jnp.array(2.0, dtype=jnp.float32),
    )
    return PRSMCCarryState(
      current_overall_state=overall_state,
      prng_key=key,
      total_swaps_attempted=jnp.array(10, dtype=jnp.int32),
      total_swaps_accepted=jnp.array(3, dtype=jnp.int32),
    )

  def test_init_success(self, sample_carry_state: PRSMCCarryState) -> None:
    """Test successful PRSMCCarryState initialization."""
    state = sample_carry_state
    assert state.current_overall_state is not None
    assert state.prng_key is not None
    chex.assert_trees_all_close(state.total_swaps_attempted, 10)
    chex.assert_trees_all_close(state.total_swaps_accepted, 3)

  def test_pytree_registration(self, sample_carry_state: PRSMCCarryState) -> None:
    """Test that PRSMCCarryState is properly registered as a PyTree."""
    state = sample_carry_state

    # Test that it can be used in JAX transformations
    def process_state(s: PRSMCCarryState) -> jax.Array:
      return s.total_swaps_attempted

    jitted_process = jax.jit(process_state)
    result = jitted_process(state)
    chex.assert_trees_all_close(result, 10)

    # Test tree flatten/unflatten
    leaves, treedef = jax.tree_util.tree_flatten(state)
    unflattened_state = jax.tree_util.tree_unflatten(treedef, leaves)

    chex.assert_trees_all_close(state.prng_key, unflattened_state.prng_key)
    chex.assert_trees_all_close(state.total_swaps_attempted, unflattened_state.total_swaps_attempted)
    chex.assert_trees_all_close(state.total_swaps_accepted, unflattened_state.total_swaps_accepted)


class TestParallelReplicaSMCOutput:
  """Test cases for ParallelReplicaSMCOutput."""

  @pytest.fixture
  def sample_output(self, valid_parallel_replica_config: ParallelReplicaConfig) -> ParallelReplicaSMCOutput:
    """Create a sample ParallelReplicaSMCOutput."""
    key = jax.random.PRNGKey(42)
    final_states = IslandState(
      key=key,
      population=jnp.array([[1, 2], [3, 4]], dtype=jnp.int32),
      beta=jnp.array(1.0, dtype=jnp.float32),
      logZ_estimate=jnp.array(3.0, dtype=jnp.float32),
      ess=jnp.array(48.0, dtype=jnp.float32),
      mean_fitness=jnp.array(3.5, dtype=jnp.float32),
    )

    return ParallelReplicaSMCOutput(
      input_config=valid_parallel_replica_config,
      final_island_states=final_states,
      swap_acceptance_rate=jnp.array(0.3, dtype=jnp.float32),
      history_mean_fitness_per_island=jnp.array([[1.0, 1.1], [1.2, 1.3]], dtype=jnp.float32),
      history_max_fitness_per_island=jnp.array([[1.5, 1.6], [1.7, 1.8]], dtype=jnp.float32),
      history_ess_per_island=jnp.array([[45.0, 46.0], [47.0, 48.0]], dtype=jnp.float32),
      history_logZ_increment_per_island=jnp.array([[0.1, 0.2], [0.3, 0.4]], dtype=jnp.float32),
      history_meta_beta=jnp.array([0.5, 1.0], dtype=jnp.float32),
      history_num_accepted_swaps=jnp.array([2.0, 3.0], dtype=jnp.float32),
      history_num_attempted_swaps=jnp.array([8.0, 10.0], dtype=jnp.float32),
    )

  def test_init_success(self, sample_output: ParallelReplicaSMCOutput) -> None:
    """Test successful ParallelReplicaSMCOutput initialization."""
    output = sample_output
    assert output.input_config is not None
    assert output.final_island_states is not None
    chex.assert_trees_all_close(output.swap_acceptance_rate, 0.3)

  def test_input_configs_property(
    self,
    sample_output: ParallelReplicaSMCOutput,
    valid_parallel_replica_config: ParallelReplicaConfig,
  ) -> None:
    """Test the input_configs property."""
    configs = sample_output.input_configs
    assert configs == valid_parallel_replica_config

  def test_per_gen_stats_metrics_property(self, sample_output: ParallelReplicaSMCOutput) -> None:
    """Test the per_gen_stats_metrics property."""
    metrics = sample_output.per_gen_stats_metrics
    expected_metrics = {
      "mean_fitness": "history_mean_fitness_per_island",
      "max_fitness": "history_max_fitness_per_island",
      "ess": "history_ess_per_island",
      "logZ_increment": "history_logZ_increment_per_island",
      "meta_beta": "history_meta_beta",
      "num_accepted_swaps": "history_num_accepted_swaps",
      "num_attempted_swaps": "history_num_attempted_swaps",
    }
    assert metrics == expected_metrics

  def test_summary_stats_metrics_property(self, sample_output: ParallelReplicaSMCOutput) -> None:
    """Test the summary_stats_metrics property."""
    metrics = sample_output.summary_stats_metrics
    expected_metrics = {
      "final_island_states": "final_island_states",
      "swap_acceptance_rate": "swap_acceptance_rate",
    }
    assert metrics == expected_metrics

  def test_output_type_name_property(self, sample_output: ParallelReplicaSMCOutput) -> None:
    """Test the output_type_name property."""
    assert sample_output.output_type_name == "ParallelReplicaSMC"

  def test_pytree_registration(self, sample_output: ParallelReplicaSMCOutput) -> None:
    """Test that ParallelReplicaSMCOutput is properly registered as a PyTree."""
    output = sample_output

    # Test that it can be used in JAX transformations
    def process_output(o: ParallelReplicaSMCOutput) -> Float:
      return o.swap_acceptance_rate

    jitted_process = jax.jit(process_output)
    result = jitted_process(output)
    chex.assert_trees_all_close(result, 0.3)

    # Test tree flatten/unflatten
    leaves, treedef = jax.tree_util.tree_flatten(output)
    unflattened_output = jax.tree_util.tree_unflatten(treedef, leaves)

    chex.assert_trees_all_close(output.swap_acceptance_rate, unflattened_output.swap_acceptance_rate)
    chex.assert_trees_all_close(
      output.history_mean_fitness_per_island,
      unflattened_output.history_mean_fitness_per_island,
    )
