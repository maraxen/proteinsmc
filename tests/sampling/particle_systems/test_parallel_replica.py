"""Tests for Parallel Replica SMC sampling functions."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import chex
import jax
import jax.numpy as jnp
import pytest
from blackjax.smc.base import SMCState
from jax import vmap

from proteinsmc.models.sampler_base import SamplerState
from proteinsmc.sampling.particle_systems.parallel_replica import (
  MigrationInfo,
  PRSMCOutput,
  migrate,
  run_prsmc_loop,
)

if TYPE_CHECKING:
  from jaxtyping import Array, PRNGKeyArray


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
    blackjax_state = SMCState(
      particles=particles,
      weights=weights,
      update_parameters=jnp.array(0.0, dtype=jnp.float32),
    )

    betas = jnp.array([0.3, 0.6, 1.0])
    mean_fitness = jnp.array([1.0, 2.0, 3.0])

    return SamplerState(
      sequence=particles,
      key=jax.random.PRNGKey(42),
      blackjax_state=blackjax_state,
      step=jnp.array(0, dtype=jnp.int32),
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
    ) -> Array:
      return jnp.ones((sequences.shape[0],))

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
    ) -> Array:
      # Return different fitness for different sequences
      return jnp.array([1.0, 2.0, 3.0])[: sequences.shape[0]]

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
    ) -> Array:
      # Return infinite fitness
      return jnp.full((sequences.shape[0],), jnp.inf)

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
    ) -> Array:
      return jnp.ones((sequences.shape[0],))

    jitted_migrate = jax.jit(
      migrate,
      static_argnums=(3, 4, 5, 6),  # n_islands, population_size, n_exchange_attempts, fitness_fn
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
      dtype=jnp.int8,  # Use int8 for protein sequences
    )
    weights = jnp.ones((n_islands, population_size)) / population_size

    # Create per-island SMCState using vmap (matching initialization_factory pattern)
    initial_weights = jnp.full(population_size, 1.0 / population_size, dtype=jnp.float32)
    # Note: update_parameters needs to be a dict for mutation_update_fn
    # mutation_rate should be batched per particle (or broadcastable)
    mutation_rates_per_particle = jnp.full(population_size, 0.1, dtype=jnp.float32)

    def create_island_state(particles_island):
      return SMCState(
        particles=particles_island,
        weights=initial_weights,
        update_parameters={"mutation_rate": mutation_rates_per_particle},
      )

    blackjax_state = vmap(create_island_state)(particles)

    betas = jnp.array([0.5, 1.0])
    mean_fitness = jnp.ones((n_islands,))
    max_fitness = jnp.ones((n_islands,))
    ess = jnp.full((n_islands,), population_size / 2.0)
    logZ_estimate = jnp.zeros((n_islands,))

    # Split PRNG key for each island (matching initialization_factory)
    island_keys = jax.random.split(jax.random.PRNGKey(42), n_islands)

    return SamplerState(
      sequence=particles,
      key=island_keys,  # Per-island keys: shape (n_islands, 2)
      blackjax_state=blackjax_state,
      step=jnp.zeros(n_islands, dtype=jnp.int32),  # Per-island step counter
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
    resampling_approach = "multinomial"
    population_size = 4
    n_islands = 2
    exchange_frequency = 1
    n_exchange_attempts = 2

    def mock_fitness_fn(
      sequence: Array,
      key: PRNGKeyArray,
      _: None,
    ) -> Array:
      """Mock fitness function that returns a scalar fitness value for a single sequence."""
      return jnp.array(1.0, dtype=jnp.float32)

    def mock_mutation_fn(
        key: PRNGKeyArray, sequence: Array, context: None
    ) -> tuple[Array, None]:
        return sequence, None

    def mock_annealing_fn(step: int) -> Array:
      return jnp.array(1.0)

    def mock_writer_callback(data: dict) -> None:
      pass

    final_state = run_prsmc_loop(
      num_steps=num_steps,
      initial_state=mock_initial_state,
      resampling_approach=resampling_approach,
      population_size=population_size,
      n_islands=n_islands,
      exchange_frequency=exchange_frequency,
      n_exchange_attempts=n_exchange_attempts,
      fitness_fn=mock_fitness_fn,
      mutation_fn=mock_mutation_fn,
      annealing_fn=mock_annealing_fn,
      writer_callback=mock_writer_callback,
    )

    assert isinstance(final_state, SamplerState)
    assert jnp.all(final_state.step == num_steps)
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

    # Use int8 dtype for sequences (protein sequence range 0-20)
    particles = jax.random.randint(
      jax.random.PRNGKey(0),
      (n_islands, population_size, seq_length),
      0,
      20,
      dtype=jnp.int8,
    )
    
    # Create batched weights and update_parameters for each island
    weights_per_island = jnp.ones((n_islands, population_size), dtype=jnp.float32) / population_size
    mutation_rates_per_island = jnp.full((n_islands, population_size), 0.1, dtype=jnp.float32)
    
    # Create BlackJAX states for each island using vmap
    blackjax_state = vmap(
      lambda p, w, mr: SMCState(
        particles=p,
        weights=w,
        update_parameters={"mutation_rate": mr},
      )
    )(particles, weights_per_island, mutation_rates_per_island)

    single_island_state = SamplerState(
      sequence=particles,
      key=jax.random.split(jax.random.PRNGKey(42), n_islands),
      blackjax_state=blackjax_state,
      step=jnp.array(0, dtype=jnp.int32),
      additional_fields={
        "beta": jnp.array([1.0]),
        "mean_fitness": jnp.ones((n_islands,)),
        "max_fitness": jnp.ones((n_islands,)),
        "ess": jnp.full((n_islands,), population_size / 2.0),
        "logZ_estimate": jnp.zeros((n_islands,)),
      },
    )

    num_steps = 2
    resampling_approach = "multinomial"
    exchange_frequency = 1
    n_exchange_attempts = 2

    def mock_fitness_fn(
      sequence: Array,
      key: PRNGKeyArray | None,
      beta: Array | None,
    ) -> Array:
      """Mock fitness function that returns a scalar fitness value and None."""
      return jnp.array(1.0, dtype=jnp.float32)

    def mock_mutation_fn(
        key: PRNGKeyArray, sequence: Array, context: None
    ) -> tuple[Array, None]:
        return sequence, None

    def mock_annealing_fn(step: int) -> float:
      """Mock annealing function."""
      return 1.0

    def mock_writer_callback(data: dict) -> None:
      pass

    final_state = run_prsmc_loop(
      num_steps=num_steps,
      initial_state=single_island_state,
      resampling_approach=resampling_approach,
      population_size=population_size,
      n_islands=n_islands,
      exchange_frequency=exchange_frequency,
      n_exchange_attempts=n_exchange_attempts,
      fitness_fn=mock_fitness_fn,
      mutation_fn=mock_mutation_fn,
      annealing_fn=mock_annealing_fn,
      writer_callback=mock_writer_callback,
    )

    assert isinstance(final_state, SamplerState)
    # For single island, step will be scalar; for multiple, it's per-island
    if hasattr(final_state.step, 'shape') and final_state.step.shape:
      assert jnp.all(final_state.step == num_steps)
    else:
      assert final_state.step == num_steps

  def test_e2e_prsmc_sampler_improves_fitness_and_migrates(
      self, mock_initial_state: SamplerState
  ) -> None:
      """An end-to-end test to validate that the PRSMC sampler improves population fitness and performs migrations."""
      num_steps = 20
      n_islands, population_size, seq_length = mock_initial_state.sequence.shape

      # A simple fitness function where lower sequence values are better
      def simple_fitness_fn(
          key: PRNGKeyArray, sequence: Array, beta: float | None
      ) -> Array:
          return -jnp.mean(sequence, axis=-1, dtype=jnp.float32)

      # A mutation function that randomly perturbs the sequence
      def mutation_fn(
          key: PRNGKeyArray, sequence: Array, context: None
      ) -> tuple[Array, None]:
          mutation = jax.random.randint(
              key, sequence.shape, -3, 3, dtype=jnp.int8
          )
          return jnp.clip(sequence + mutation, 0, 19), None

      def mock_annealing_fn(step: int) -> Array:
          return jnp.array(1.0)

      # A writer callback to store the number of accepted swaps
      accepted_swaps = []

      def mock_writer_callback(data: dict) -> None:
          accepted_swaps.append(data["num_accepted_swaps"])

      # Calculate initial mean fitness
      initial_fitness = jax.vmap(simple_fitness_fn, in_axes=(None, 0, None))(
          None, mock_initial_state.sequence.reshape(-1, seq_length), None
      )
      initial_mean_fitness = jnp.mean(initial_fitness)

      # Run the PRSMC loop
      final_state = run_prsmc_loop(
          num_steps=num_steps,
          initial_state=mock_initial_state,
          resampling_approach="systematic",
          population_size=population_size,
          n_islands=n_islands,
          exchange_frequency=1,
          n_exchange_attempts=5,
          fitness_fn=simple_fitness_fn,
          mutation_fn=mutation_fn,
          annealing_fn=mock_annealing_fn,
          writer_callback=mock_writer_callback,
      )

      # Calculate final mean fitness
      final_fitness = jax.vmap(simple_fitness_fn, in_axes=(None, 0, None))(
          None, final_state.sequence.reshape(-1, seq_length), None
      )
      final_mean_fitness = jnp.mean(final_fitness)

      # Assert that the average fitness has improved
      assert final_mean_fitness > initial_mean_fitness
      # Assert that replica exchange has occurred
      assert sum(accepted_swaps) > 0



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
    blackjax_state = SMCState(
      particles=particles,
      weights=weights,
      update_parameters=jnp.array(0.0, dtype=jnp.float32),
    )

    state = SamplerState(
      sequence=particles,
      key=jax.random.PRNGKey(42),
      blackjax_state=blackjax_state,
      step=jnp.array(1, dtype=jnp.int32),
      additional_fields={"beta": jnp.array([0.5, 1.0])},
    )

    from blackjax.smc.base import SMCInfo

    info = SMCInfo(
      log_likelihood_increment=jnp.zeros((n_islands,)),
      ancestors=jnp.zeros((n_islands, population_size), dtype=jnp.int32),
      update_info={},
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
