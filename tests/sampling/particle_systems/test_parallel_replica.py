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

from proteinsmc.models.sampler_base import SamplerOutput, SamplerState
from proteinsmc.sampling.particle_systems.parallel_replica import (
    migrate,
    run_prsmc_loop,
)
from proteinsmc.sampling.particle_systems.smc import UpdateInfo

if TYPE_CHECKING:
    from jaxtyping import Array, Float, Int, PRNGKeyArray


class TestMigrate:
    """Test the migrate function for replica exchange."""

    @pytest.fixture
    def mock_island_state(self) -> SamplerState:
        """Create a mock island state for testing."""
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
        blackjax_state = vmap(
            lambda p, w: SMCState(particles=p, weights=w, update_parameters={})
        )(particles, weights)

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
        """Test basic migrate functionality."""
        meta_beta = jnp.array(0.5)
        key = jax.random.PRNGKey(123)
        n_islands = 3
        population_size = 4
        n_exchange_attempts = 5

        def mock_fitness_fn(
            key: PRNGKeyArray,
            sequences: Array,
            beta: Float | None,
        ) -> Array:
            return jnp.ones((sequences.shape[0],))

        (
            updated_state,
            num_accepted,
            island_from,
            island_to,
            particle_from,
            particle_to,
            accepted,
            log_ratio,
        ) = migrate(
            mock_island_state,
            meta_beta,
            key,
            n_islands,
            population_size,
            n_exchange_attempts,
            mock_fitness_fn, # type:ignore 
        )

        # Check output structure
        assert isinstance(updated_state, SamplerState)
        chex.assert_type(num_accepted, jnp.int32)

        # Check migration info shapes (now individual arrays instead of MigrationInfo)
        chex.assert_shape(island_from, (n_exchange_attempts,))
        chex.assert_shape(island_to, (n_exchange_attempts,))
        chex.assert_shape(particle_from, (n_exchange_attempts,))
        chex.assert_shape(particle_to, (n_exchange_attempts,))
        chex.assert_shape(accepted, (n_exchange_attempts,))
        chex.assert_shape(log_ratio, (n_exchange_attempts,))


@pytest.fixture
def mock_prsmc_initial_state() -> SamplerState:
    """Create a mock initial state for PRSMC loop."""
    n_islands = 2
    population_size = 4
    seq_length = 3

    particles = jax.random.randint(
        jax.random.PRNGKey(0),
        (n_islands, population_size, seq_length),
        0,
        20,
        dtype=jnp.int8,
    )
    weights = jnp.ones((n_islands, population_size)) / population_size

    def create_island_state(particles_island, weights_island):
        return SMCState(
            particles=particles_island,
            weights=weights_island,
            update_parameters={},
        )

    blackjax_state = vmap(create_island_state)(particles, weights)

    betas = jnp.array([0.5, 1.0])
    mean_fitness = jnp.ones((n_islands,))
    max_fitness = jnp.ones((n_islands,))
    ess = jnp.full((n_islands,), population_size / 2.0)
    logZ_estimate = jnp.zeros((n_islands,))

    island_keys = jax.random.split(jax.random.PRNGKey(42), n_islands)

    return SamplerState(
        sequence=particles,
        key=island_keys,
        blackjax_state=blackjax_state,
        step=jnp.array(0, dtype=jnp.int32),
        additional_fields={
            "beta": betas,
            "mean_fitness": mean_fitness,
            "max_fitness": max_fitness,
            "ess": ess,
            "logZ_estimate": logZ_estimate,
        },
    )


class TestRunPRSMCLoop:
    """Test the run_prsmc_loop function."""

    def test_run_prsmc_loop_basic(
        self, mock_prsmc_initial_state: SamplerState
    ) -> None:
        """Test basic run_prsmc_loop functionality."""
        num_steps = 2
        resampling_approach = "multinomial"
        population_size = 4
        n_islands = 2
        exchange_frequency = 1
        n_exchange_attempts = 2

        def mock_fitness_fn(
            key: PRNGKeyArray, sequence: Array, beta: Float | None
        ) -> Array:
            return jnp.ones((), dtype=jnp.float32)

        def mock_mutation_fn(
            key: PRNGKeyArray, sequence: Array, context: None
        ) -> tuple[Array, UpdateInfo]:
            return sequence, UpdateInfo()

        def mock_annealing_fn(current_step: int, _context: None) -> Array:
            return jnp.array(1.0)

        def mock_writer_callback(data: dict) -> None:
            pass

        final_state = run_prsmc_loop(
            num_steps=num_steps,
            initial_state=mock_prsmc_initial_state,
            resampling_approach=resampling_approach,
            population_size=population_size,
            n_islands=n_islands,
            exchange_frequency=exchange_frequency,
            n_exchange_attempts=n_exchange_attempts,
            fitness_fn=mock_fitness_fn,  # type:ignore
            mutation_fn=mock_mutation_fn,  # type:ignore
            annealing_fn=mock_annealing_fn,  # type:ignore
            writer_callback=mock_writer_callback,
        )

        assert isinstance(final_state, SamplerState)
        assert final_state.step == num_steps
        chex.assert_shape(
            final_state.sequence,
            (
                n_islands,
                population_size,
                mock_prsmc_initial_state.sequence.shape[-1],
            ),
        )

    def test_run_prsmc_loop_end_to_end(
        self, mock_prsmc_initial_state: SamplerState
    ) -> None:
        """Test the run_prsmc_loop with a simple end-to-end case."""
        num_steps = 5
        resampling_approach = "systematic"
        population_size = 4
        n_islands = 2
        exchange_frequency = 2
        n_exchange_attempts = 5

        def fitness_fn(key: PRNGKeyArray, sequence: Array, beta: Float) -> Array:
            # A simple fitness function that prefers sequences with higher sums
            return jnp.sum(sequence, axis=-1) * beta

        def mutation_fn(
            key: PRNGKeyArray, sequence: Array, context: None
        ) -> tuple[Array, UpdateInfo]:
            # A simple mutation that adds 1 to a random position
            pos = jax.random.randint(
                key, (1,), 0, sequence.shape[-1], dtype=jnp.int32
            )[0]
            return sequence.at[..., pos].set(sequence[..., pos] + 1), UpdateInfo()

        def annealing_fn(current_step: Int, _context: None) -> Float:
            return current_step / num_steps

        def writer_callback(output: SamplerOutput) -> None:
            # Check that the output is of the correct type
            assert isinstance(output, SamplerOutput)
            assert hasattr(output, "sequences")
            assert hasattr(output, "num_attempted_swaps")

        final_state = run_prsmc_loop(
            num_steps=num_steps,
            initial_state=mock_prsmc_initial_state,
            resampling_approach=resampling_approach,
            population_size=population_size,
            n_islands=n_islands,
            exchange_frequency=exchange_frequency,
            n_exchange_attempts=n_exchange_attempts,
            fitness_fn=fitness_fn,
            mutation_fn=mutation_fn,
            annealing_fn=annealing_fn,
            writer_callback=writer_callback,
        )

        assert isinstance(final_state, SamplerState)
        assert final_state.step == num_steps

        # Check that the fitness of the particles has generally increased
        initial_fitness = jnp.mean(
            vmap(vmap(fitness_fn, in_axes=(None, 0, None)), in_axes=(None, 0, 0))(
                jax.random.PRNGKey(0),
                mock_prsmc_initial_state.sequence,
                jnp.array([1.0, 1.0]),
            )
        )
        final_fitness = jnp.mean(
            vmap(vmap(fitness_fn, in_axes=(None, 0, None)), in_axes=(None, 0, 0))(
                jax.random.PRNGKey(1), final_state.sequence, jnp.array([1.0, 1.0])
            )
        )
        assert final_fitness > initial_fitness


# NOTE: TestMigrationInfo and TestPRSMCOutput classes have been removed
# as MigrationInfo and PRSMCOutput data structures have been replaced
# with the unified SamplerOutput structure. Migration data is now stored
# as individual array fields in SamplerOutput rather than a nested structure.
