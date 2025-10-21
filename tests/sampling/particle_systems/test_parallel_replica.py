"""Tests for Parallel Replica SMC sampling functions."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import chex
import jax
import jax.numpy as jnp
import pytest
from blackjax.smc.base import SMCState

from proteinsmc.models.sampler_base import SamplerState
from proteinsmc.sampling.particle_systems.parallel_replica import (
  MigrationInfo,
  PRSMCOutput,
  migrate,
  mutation_update_fn,
  run_prsmc_loop,
  weight_fn,
)

if TYPE_CHECKING:
  from jaxtyping import Array, PRNGKeyArray


class TestMutationUpdateFn:
  """Test the mutation_update_fn for parallel replica SMC."""

  def test_mutation_update_fn_protein(self) -> None:
    """Test mutation_update_fn with protein sequences.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If mutation output shape or type is incorrect.

    Example:
        >>> test_mutation_update_fn_protein()

    """
    key = jax.random.PRNGKey(42)
    n_particles = 10
    seq_length = 5
    keys = jax.random.split(key, n_particles)
    sequences = jax.random.randint(key, (n_particles, seq_length), 0, 20)
    mutation_rates = jnp.full((n_particles,), 0.1)
    update_parameters = {"mutation_rate": mutation_rates}

    mutated, _ = mutation_update_fn(keys, sequences, update_parameters, q_states=20)

    chex.assert_shape(mutated, (n_particles, seq_length))
    chex.assert_type(mutated, jnp.int32)
    # At least some mutations should have occurred (probabilistically)
    assert not jnp.array_equal(mutated, sequences)

  def test_mutation_update_fn_nucleotide(self) -> None:
    """Test mutation_update_fn with nucleotide sequences.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If mutation output shape or type is incorrect.

    Example:
        >>> test_mutation_update_fn_nucleotide()

    """
    key = jax.random.PRNGKey(123)
    n_particles = 8
    seq_length = 10
    keys = jax.random.split(key, n_particles)
    sequences = jax.random.randint(key, (n_particles, seq_length), 0, 4)
    mutation_rates = jnp.full((n_particles,), 0.2)
    update_parameters = {"mutation_rate": mutation_rates}

    mutated, _ = mutation_update_fn(keys, sequences, update_parameters, q_states=4)

    chex.assert_shape(mutated, (n_particles, seq_length))
    chex.assert_type(mutated, jnp.int32)
    # Verify values are within valid nucleotide range
    assert jnp.all((mutated >= 0) & (mutated < 4))

  def test_mutation_update_fn_zero_rate(self) -> None:
    """Test mutation_update_fn with zero mutation rate.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If sequences change despite zero mutation rate.

    Example:
        >>> test_mutation_update_fn_zero_rate()

    """
    key = jax.random.PRNGKey(999)
    n_particles = 5
    seq_length = 8
    keys = jax.random.split(key, n_particles)
    sequences = jax.random.randint(key, (n_particles, seq_length), 0, 20)
    mutation_rates = jnp.zeros((n_particles,))
    update_parameters = {"mutation_rate": mutation_rates}

    mutated, _ = mutation_update_fn(keys, sequences, update_parameters, q_states=20)

    # With zero mutation rate, sequences should remain unchanged
    chex.assert_trees_all_equal(mutated, sequences)


class TestWeightFn:
  """Test the weight_fn for SMC step weighting."""

  def test_weight_fn_finite_fitness(self) -> None:
    """Test weight_fn with finite fitness values.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If weight calculation is incorrect.

    Example:
        >>> test_weight_fn_finite_fitness()

    """
    sequence = jnp.array([1, 2, 3, 4, 5])
    fitness_value = jnp.array(2.0)
    beta = jnp.array(0.5)

    def mock_fitness_fn(seq: Array) -> tuple[Array, None]:
      return fitness_value, None

    weight = weight_fn(sequence, mock_fitness_fn, beta)

    expected_weight = beta * fitness_value
    chex.assert_trees_all_close(weight, expected_weight, rtol=1e-5)
    assert weight.dtype == jnp.bfloat16

  def test_weight_fn_negative_infinity_fitness(self) -> None:
    """Test weight_fn with negative infinity fitness.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If weight is not negative infinity.

    Example:
        >>> test_weight_fn_negative_infinity_fitness()

    """
    sequence = jnp.array([1, 2, 3])
    fitness_value = jnp.array(-jnp.inf)
    beta = jnp.array(1.0)

    def mock_fitness_fn(seq: Array) -> tuple[Array, None]:
      return fitness_value, None

    weight = weight_fn(sequence, mock_fitness_fn, beta)

    assert jnp.isneginf(weight)

  def test_weight_fn_zero_beta(self) -> None:
    """Test weight_fn with zero beta (cold temperature).

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If weight is not zero.

    Example:
        >>> test_weight_fn_zero_beta()

    """
    sequence = jnp.array([1, 2, 3])
    fitness_value = jnp.array(5.0)
    beta = jnp.array(0.0)

    def mock_fitness_fn(seq: Array) -> tuple[Array, None]:
      return fitness_value, None

    weight = weight_fn(sequence, mock_fitness_fn, beta)

    chex.assert_trees_all_close(weight, jnp.array(0.0, dtype=jnp.bfloat16), rtol=1e-5)

  def test_weight_fn_jit_compatible(self) -> None:
    """Test that weight_fn is JIT-compatible.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If JIT compilation fails.

    Example:
        >>> test_weight_fn_jit_compatible()

    """
    sequence = jnp.array([1, 2, 3, 4, 5])
    fitness_value = jnp.array(3.0)
    beta = jnp.array(0.8)

    def mock_fitness_fn(seq: Array) -> tuple[Array, None]:
      return fitness_value, None

    jitted_weight_fn = jax.jit(weight_fn, static_argnums=1)
    weight = jitted_weight_fn(sequence, mock_fitness_fn, beta)

    expected_weight = beta * fitness_value
    chex.assert_trees_all_close(weight, expected_weight, rtol=1e-5)


class TestMigrate:
  """Test the migrate function for replica exchange."""

  @pytest.fixture
  def mock_island_state(self) -> SamplerState:
    """Create a mock island state for testing.

    Args:
        None

    Returns:
        SamplerState: Mock state with 3 islands, 4 particles each.

    Raises:
        None

    Example:
        >>> mock_island_state()

    """
    n_islands = 3
    population_size = 4
    seq_length = 5

    particles = jax.random.randint(
      jax.random.PRNGKey(0),
      (n_islands, population_size, seq_length),
      0,
      20,
    )
    weights = jnp.ones((n_islands, population_size)) / population_size
    blackjax_state = SMCState(particles=particles, weights=weights)

    betas = jnp.array([0.3, 0.6, 1.0])
    mean_fitness = jnp.array([1.0, 2.0, 3.0])

    return SamplerState(
      sequence=particles,
      fitness=jnp.ones((n_islands, population_size)),
      key=jax.random.PRNGKey(42),
      blackjax_state=blackjax_state,
      step=0,
      additional_fields={
        "beta": betas,
        "mean_fitness": mean_fitness,
      },
    )

  def test_migrate_basic_functionality(self, mock_island_state: SamplerState) -> None:
    """Test basic migrate functionality.

    Args:
        mock_island_state: Fixture providing mock island state.

    Returns:
        None

    Raises:
        AssertionError: If migration output structure is incorrect.

    Example:
        >>> test_migrate_basic_functionality(mock_island_state)

    """
    meta_beta = jnp.array(0.5)
    key = jax.random.PRNGKey(123)
    n_islands = 3
    population_size = 4
    n_exchange_attempts = 5

    def mock_fitness_fn(
      sequences: Array,
      key: PRNGKeyArray,
      _: None,
    ) -> tuple[Array, None]:
      return jnp.ones((sequences.shape[0],)), None

    updated_state, num_accepted, migration_info = migrate(
      mock_island_state,
      meta_beta,
      key,
      n_islands,
      population_size,
      n_exchange_attempts,
      mock_fitness_fn,
    )

    # Check output structure
    assert isinstance(updated_state, SamplerState)
    chex.assert_type(num_accepted, jnp.int32)
    assert isinstance(migration_info, MigrationInfo)

    # Check migration info shapes
    chex.assert_shape(migration_info.island_from, (n_exchange_attempts,))
    chex.assert_shape(migration_info.island_to, (n_exchange_attempts,))
    chex.assert_shape(migration_info.particle_idx_from, (n_exchange_attempts,))
    chex.assert_shape(migration_info.particle_idx_to, (n_exchange_attempts,))
    chex.assert_shape(migration_info.accepted, (n_exchange_attempts,))
    chex.assert_shape(migration_info.log_acceptance_ratio, (n_exchange_attempts,))

  def test_migrate_acceptance_ratio_logic(
    self,
    mock_island_state: SamplerState,
  ) -> None:
    """Test that acceptance ratio follows Metropolis-Hastings criterion.

    Args:
        mock_island_state: Fixture providing mock island state.

    Returns:
        None

    Raises:
        AssertionError: If acceptance probability logic is incorrect.

    Example:
        >>> test_migrate_acceptance_ratio_logic(mock_island_state)

    """
    meta_beta = jnp.array(1.0)
    key = jax.random.PRNGKey(456)
    n_islands = 3
    population_size = 4
    n_exchange_attempts = 10

    # Create fitness function that returns known values
    def mock_fitness_fn(
      sequences: Array,
      key: PRNGKeyArray,
      _: None,
    ) -> tuple[Array, None]:
      # Return different fitness for different sequences
      return jnp.array([1.0, 2.0, 3.0])[: sequences.shape[0]], None

    updated_state, num_accepted, migration_info = migrate(
      mock_island_state,
      meta_beta,
      key,
      n_islands,
      population_size,
      n_exchange_attempts,
      mock_fitness_fn,
    )

    # Check that acceptance ratios are finite (no NaN or Inf for valid swaps)
    valid_ratios = migration_info.log_acceptance_ratio[
      ~jnp.isneginf(migration_info.log_acceptance_ratio)
    ]
    assert jnp.all(jnp.isfinite(valid_ratios))

    # Check that num_accepted is between 0 and n_exchange_attempts
    assert 0 <= num_accepted <= n_exchange_attempts

  def test_migrate_infinite_fitness_rejection(
    self,
    mock_island_state: SamplerState,
  ) -> None:
    """Test that infinite fitness leads to rejection.

    Args:
        mock_island_state: Fixture providing mock island state.

    Returns:
        None

    Raises:
        AssertionError: If infinite fitness swaps are accepted.

    Example:
        >>> test_migrate_infinite_fitness_rejection(mock_island_state)

    """
    meta_beta = jnp.array(0.5)
    key = jax.random.PRNGKey(789)
    n_islands = 3
    population_size = 4
    n_exchange_attempts = 5

    def mock_fitness_fn(
      sequences: Array,
      key: PRNGKeyArray,
      _: None,
    ) -> tuple[Array, None]:
      # Return infinite fitness
      return jnp.full((sequences.shape[0],), jnp.inf), None

    updated_state, num_accepted, migration_info = migrate(
      mock_island_state,
      meta_beta,
      key,
      n_islands,
      population_size,
      n_exchange_attempts,
      mock_fitness_fn,
    )

    # All swaps with infinite fitness should be rejected
    # log_acceptance_ratio should be -inf
    assert jnp.all(jnp.isneginf(migration_info.log_acceptance_ratio))
    assert num_accepted == 0

  def test_migrate_jit_compatible(self, mock_island_state: SamplerState) -> None:
    """Test that migrate is JIT-compatible.

    Args:
        mock_island_state: Fixture providing mock island state.

    Returns:
        None

    Raises:
        AssertionError: If JIT compilation fails.

    Example:
        >>> test_migrate_jit_compatible(mock_island_state)

    """
    meta_beta = jnp.array(0.5)
    key = jax.random.PRNGKey(999)
    n_islands = 3
    population_size = 4
    n_exchange_attempts = 5

    def mock_fitness_fn(
      sequences: Array,
      key: PRNGKeyArray,
      _: None,
    ) -> tuple[Array, None]:
      return jnp.ones((sequences.shape[0],)), None

    jitted_migrate = jax.jit(
      migrate,
      static_argnums=(3, 4, 5),  # n_islands, population_size, n_exchange_attempts
    )

    updated_state, num_accepted, migration_info = jitted_migrate(
      mock_island_state,
      meta_beta,
      key,
      n_islands,
      population_size,
      n_exchange_attempts,
      mock_fitness_fn,
    )

    assert isinstance(updated_state, SamplerState)
    assert isinstance(migration_info, MigrationInfo)


class TestRunPRSMCLoop:
  """Test the run_prsmc_loop function."""

  @pytest.fixture
  def mock_initial_state(self) -> SamplerState:
    """Create a mock initial state for PRSMC loop.

    Args:
        None

    Returns:
        SamplerState: Mock initial state with 2 islands, 4 particles each.

    Raises:
        None

    Example:
        >>> mock_initial_state()

    """
    n_islands = 2
    population_size = 4
    seq_length = 3

    particles = jax.random.randint(
      jax.random.PRNGKey(0),
      (n_islands, population_size, seq_length),
      0,
      20,
    )
    weights = jnp.ones((n_islands, population_size)) / population_size
    blackjax_state = SMCState(particles=particles, weights=weights)

    betas = jnp.array([0.5, 1.0])
    mean_fitness = jnp.ones((n_islands,))
    max_fitness = jnp.ones((n_islands,))
    ess = jnp.full((n_islands,), population_size / 2.0)
    logZ_estimate = jnp.zeros((n_islands,))

    return SamplerState(
      sequence=particles,
      fitness=jnp.ones((n_islands, population_size)),
      key=jax.random.PRNGKey(42),
      blackjax_state=blackjax_state,
      step=0,
      additional_fields={
        "beta": betas,
        "mean_fitness": mean_fitness,
        "max_fitness": max_fitness,
        "ess": ess,
        "logZ_estimate": logZ_estimate,
      },
    )

  def test_run_prsmc_loop_basic(self, mock_initial_state: SamplerState) -> None:
    """Test basic run_prsmc_loop functionality.

    Args:
        mock_initial_state: Fixture providing mock initial state.

    Returns:
        None

    Raises:
        AssertionError: If loop execution fails or output is incorrect.

    Example:
        >>> test_run_prsmc_loop_basic(mock_initial_state)

    """
    num_steps = 2
    sequence_type = "protein"
    resampling_approach = "multinomial"
    population_size = 4
    n_islands = 2
    exchange_frequency = 1
    n_exchange_attempts = 2

    def mock_fitness_fn(
      sequences: Array,
      key: PRNGKeyArray,
      _: None,
    ) -> tuple[Array, None]:
      return jnp.ones((sequences.shape[0], sequences.shape[1])), None

    def mock_annealing_fn(step: int) -> Array:
      return jnp.array(1.0)

    def mock_writer_callback(data: dict) -> None:
      pass

    final_state = run_prsmc_loop(
      num_steps=num_steps,
      initial_state=mock_initial_state,
      sequence_type=sequence_type,
      resampling_approach=resampling_approach,
      population_size=population_size,
      n_islands=n_islands,
      exchange_frequency=exchange_frequency,
      n_exchange_attempts=n_exchange_attempts,
      fitness_fn=mock_fitness_fn,
      annealing_fn=mock_annealing_fn,
      writer_callback=mock_writer_callback,
    )

    assert isinstance(final_state, SamplerState)
    assert final_state.step == num_steps
    chex.assert_shape(
      final_state.sequence,
      (n_islands, population_size, mock_initial_state.sequence.shape[-1]),
    )

  def test_run_prsmc_loop_no_exchange(self, mock_initial_state: SamplerState) -> None:
    """Test run_prsmc_loop with single island (no exchange).

    Args:
        mock_initial_state: Fixture providing mock initial state.

    Returns:
        None

    Raises:
        AssertionError: If loop execution fails with single island.

    Example:
        >>> test_run_prsmc_loop_no_exchange(mock_initial_state)

    """
    # Modify state to have single island
    n_islands = 1
    population_size = 4
    seq_length = 3

    particles = jax.random.randint(
      jax.random.PRNGKey(0),
      (n_islands, population_size, seq_length),
      0,
      20,
    )
    weights = jnp.ones((n_islands, population_size)) / population_size
    blackjax_state = SMCState(particles=particles, weights=weights)

    single_island_state = SamplerState(
      sequence=particles,
      fitness=jnp.ones((n_islands, population_size)),
      key=jax.random.PRNGKey(42),
      blackjax_state=blackjax_state,
      step=0,
      additional_fields={
        "beta": jnp.array([1.0]),
        "mean_fitness": jnp.ones((n_islands,)),
        "max_fitness": jnp.ones((n_islands,)),
        "ess": jnp.full((n_islands,), population_size / 2.0),
        "logZ_estimate": jnp.zeros((n_islands,)),
      },
    )

    num_steps = 2
    sequence_type = "protein"
    resampling_approach = "multinomial"
    exchange_frequency = 1
    n_exchange_attempts = 2

    def mock_fitness_fn(
      sequences: Array,
      key: PRNGKeyArray,
      _: None,
    ) -> tuple[Array, None]:
      return jnp.ones((sequences.shape[0], sequences.shape[1])), None

    def mock_annealing_fn(step: int) -> Array:
      return jnp.array(1.0)

    def mock_writer_callback(data: dict) -> None:
      pass

    final_state = run_prsmc_loop(
      num_steps=num_steps,
      initial_state=single_island_state,
      sequence_type=sequence_type,
      resampling_approach=resampling_approach,
      population_size=population_size,
      n_islands=n_islands,
      exchange_frequency=exchange_frequency,
      n_exchange_attempts=n_exchange_attempts,
      fitness_fn=mock_fitness_fn,
      annealing_fn=mock_annealing_fn,
      writer_callback=mock_writer_callback,
    )

    assert isinstance(final_state, SamplerState)
    assert final_state.step == num_steps

  def test_run_prsmc_loop_nucleotide_sequence(
    self,
    mock_initial_state: SamplerState,
  ) -> None:
    """Test run_prsmc_loop with nucleotide sequences.

    Args:
        mock_initial_state: Fixture providing mock initial state.

    Returns:
        None

    Raises:
        AssertionError: If loop fails with nucleotide sequences.

    Example:
        >>> test_run_prsmc_loop_nucleotide_sequence(mock_initial_state)

    """
    # Modify state for nucleotide sequences
    n_islands = 2
    population_size = 4
    seq_length = 6

    particles = jax.random.randint(
      jax.random.PRNGKey(0),
      (n_islands, population_size, seq_length),
      0,
      4,  # Nucleotide range
    )
    weights = jnp.ones((n_islands, population_size)) / population_size
    blackjax_state = SMCState(particles=particles, weights=weights)

    nucleotide_state = SamplerState(
      sequence=particles,
      fitness=jnp.ones((n_islands, population_size)),
      key=jax.random.PRNGKey(42),
      blackjax_state=blackjax_state,
      step=0,
      additional_fields={
        "beta": jnp.array([0.5, 1.0]),
        "mean_fitness": jnp.ones((n_islands,)),
        "max_fitness": jnp.ones((n_islands,)),
        "ess": jnp.full((n_islands,), population_size / 2.0),
        "logZ_estimate": jnp.zeros((n_islands,)),
      },
    )

    num_steps = 1
    sequence_type = "nucleotide"
    resampling_approach = "multinomial"
    exchange_frequency = 1
    n_exchange_attempts = 2

    def mock_fitness_fn(
      sequences: Array,
      key: PRNGKeyArray,
      _: None,
    ) -> tuple[Array, None]:
      return jnp.ones((sequences.shape[0], sequences.shape[1])), None

    def mock_annealing_fn(step: int) -> Array:
      return jnp.array(1.0)

    def mock_writer_callback(data: dict) -> None:
      pass

    final_state = run_prsmc_loop(
      num_steps=num_steps,
      initial_state=nucleotide_state,
      sequence_type=sequence_type,
      resampling_approach=resampling_approach,
      population_size=population_size,
      n_islands=n_islands,
      exchange_frequency=exchange_frequency,
      n_exchange_attempts=n_exchange_attempts,
      fitness_fn=mock_fitness_fn,
      annealing_fn=mock_annealing_fn,
      writer_callback=mock_writer_callback,
    )

    assert isinstance(final_state, SamplerState)
    # Verify output sequences are still in nucleotide range
    assert jnp.all((final_state.sequence >= 0) & (final_state.sequence < 4))


class TestMigrationInfo:
  """Test the MigrationInfo dataclass."""

  def test_migration_info_creation(self) -> None:
    """Test MigrationInfo creation.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If creation fails.

    Example:
        >>> test_migration_info_creation()

    """
    n_attempts = 5
    migration_info = MigrationInfo(
      island_from=jnp.zeros(n_attempts, dtype=jnp.int32),
      island_to=jnp.ones(n_attempts, dtype=jnp.int32),
      particle_idx_from=jnp.arange(n_attempts, dtype=jnp.int32),
      particle_idx_to=jnp.arange(n_attempts, dtype=jnp.int32),
      accepted=jnp.array([True, False, True, False, True]),
      log_acceptance_ratio=jnp.zeros(n_attempts, dtype=jnp.float32),
    )

    chex.assert_shape(migration_info.island_from, (n_attempts,))
    chex.assert_shape(migration_info.island_to, (n_attempts,))
    chex.assert_shape(migration_info.particle_idx_from, (n_attempts,))
    chex.assert_shape(migration_info.particle_idx_to, (n_attempts,))
    chex.assert_shape(migration_info.accepted, (n_attempts,))
    chex.assert_shape(migration_info.log_acceptance_ratio, (n_attempts,))

  def test_migration_info_types(self) -> None:
    """Test MigrationInfo field types.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If field types are incorrect.

    Example:
        >>> test_migration_info_types()

    """
    n_attempts = 3
    migration_info = MigrationInfo(
      island_from=jnp.array([0, 1, 2], dtype=jnp.int32),
      island_to=jnp.array([1, 2, 0], dtype=jnp.int32),
      particle_idx_from=jnp.array([0, 1, 2], dtype=jnp.int32),
      particle_idx_to=jnp.array([2, 1, 0], dtype=jnp.int32),
      accepted=jnp.array([True, True, False]),
      log_acceptance_ratio=jnp.array([0.5, 0.3, -1.0], dtype=jnp.float32),
    )

    chex.assert_type(migration_info.island_from, jnp.int32)
    chex.assert_type(migration_info.island_to, jnp.int32)
    chex.assert_type(migration_info.particle_idx_from, jnp.int32)
    chex.assert_type(migration_info.particle_idx_to, jnp.int32)
    chex.assert_type(migration_info.accepted, bool)
    chex.assert_type(migration_info.log_acceptance_ratio, jnp.float32)


class TestPRSMCOutput:
  """Test the PRSMCOutput dataclass."""

  def test_prsmc_output_creation(self) -> None:
    """Test PRSMCOutput creation from sampling/particle_systems module.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If creation fails.

    Example:
        >>> test_prsmc_output_creation()

    """
    n_islands = 2
    population_size = 4
    seq_length = 3
    n_exchange_attempts = 5

    particles = jax.random.randint(
      jax.random.PRNGKey(0),
      (n_islands, population_size, seq_length),
      0,
      20,
    )
    weights = jnp.ones((n_islands, population_size)) / population_size
    blackjax_state = SMCState(particles=particles, weights=weights)

    state = SamplerState(
      sequence=particles,
      fitness=jnp.ones((n_islands, population_size)),
      key=jax.random.PRNGKey(42),
      blackjax_state=blackjax_state,
      step=1,
      additional_fields={"beta": jnp.array([0.5, 1.0])},
    )

    from blackjax.smc.base import SMCInfo

    info = SMCInfo(
      log_likelihood_increment=jnp.zeros((n_islands,)),
    )

    migration_info = MigrationInfo(
      island_from=jnp.zeros(n_exchange_attempts, dtype=jnp.int32),
      island_to=jnp.ones(n_exchange_attempts, dtype=jnp.int32),
      particle_idx_from=jnp.zeros(n_exchange_attempts, dtype=jnp.int32),
      particle_idx_to=jnp.zeros(n_exchange_attempts, dtype=jnp.int32),
      accepted=jnp.zeros(n_exchange_attempts, dtype=jnp.bool_),
      log_acceptance_ratio=jnp.zeros(n_exchange_attempts, dtype=jnp.float32),
    )

    output = PRSMCOutput(
      state=state,
      info=info,
      num_attempted_swaps=jnp.array(n_exchange_attempts, dtype=jnp.int32),
      num_accepted_swaps=jnp.array(2, dtype=jnp.int32),
      migration_info=migration_info,
    )

    assert isinstance(output.state, SamplerState)
    assert isinstance(output.info, SMCInfo)
    assert output.num_attempted_swaps == n_exchange_attempts
    assert output.num_accepted_swaps == 2
    assert isinstance(output.migration_info, MigrationInfo)
