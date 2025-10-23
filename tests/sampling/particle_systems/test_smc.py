"""Tests for SMC sampling functions."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import chex
import jax
import jax.numpy as jnp
import pytest
from blackjax.smc.base import SMCState
from jax.flatten_util import ravel_pytree

from proteinsmc.models.sampler_base import SamplerState
from proteinsmc.sampling.particle_systems.smc import (
    create_smc_loop_func,
    resample,
    run_smc_loop,
)

if TYPE_CHECKING:
    from jaxtyping import Array, Float, PRNGKeyArray

    from proteinsmc.models.mutation import MutationFn
    from proteinsmc.models.smc import SMCAlgorithmType


class TestResample:
    """Test the resample function."""

    @pytest.mark.parametrize(
        "resampling_approach",
        ["systematic", "multinomial", "stratified", "residual"],
    )
    def test_resample_output_shape_and_type(
        self, resampling_approach: str
    ) -> None:
        """Test that resample returns ancestors of the correct shape and type."""
        key = jax.random.PRNGKey(0)
        n_particles = 10
        weights = jnp.full(n_particles, 1.0 / n_particles)
        ancestors = resample(resampling_approach, key, weights, n_particles)
        chex.assert_shape(ancestors, (n_particles,))
        chex.assert_type(ancestors, jnp.int32)

    def test_resample_invalid_approach_raises_error(self) -> None:
        """Test that an invalid resampling approach raises a ValueError."""
        key = jax.random.PRNGKey(0)
        weights = jnp.array([0.5, 0.5])
        with pytest.raises(ValueError, match="Unknown resampling approach"):
            resample("invalid_method", key, weights, 2)


class TestCreateSMCLoopFunc:
    """Test the create_smc_loop_func factory."""

    @pytest.mark.parametrize(
        "algorithm", ["BaseSMC", "AnnealedSMC", "ParallelReplicaSMC"]
    )
    def test_create_smc_loop_func_returns_callable(
        self, algorithm: SMCAlgorithmType
    ) -> None:
        """Test that the factory returns a callable for valid algorithms."""
        mutation_fn = lambda key, state, **kwargs: (state, None)
        loop_fn = create_smc_loop_func(
            algorithm=algorithm,
            resampling_approach="multinomial",
            mutation_fn=mutation_fn,
        )
        assert callable(loop_fn)

    def test_create_smc_loop_func_adaptive_tempered_raises_error(self) -> None:
        """Test that 'AdaptiveTemperedSMC' raises a NotImplementedError."""
        mutation_fn = lambda key, state, **kwargs: (state, None)
        with pytest.raises(NotImplementedError):
            create_smc_loop_func(
                algorithm="AdaptiveTemperedSMC",
                resampling_approach="multinomial",
                mutation_fn=mutation_fn,
            )

    def test_create_smc_loop_func_invalid_algorithm_raises_error(self) -> None:
        """Test that an invalid algorithm name raises a NotImplementedError."""
        mutation_fn = lambda key, state, **kwargs: (state, None)
        with pytest.raises(NotImplementedError):
            create_smc_loop_func(
                algorithm="InvalidAlgorithm",
                resampling_approach="multinomial",
                mutation_fn=mutation_fn,
            )


class TestRunSMCLoop:
    """Test the main run_smc_loop function."""

    @pytest.fixture
    def mock_initial_state(self) -> SamplerState:
        """Create a mock initial state for the SMC loop."""
        n_particles = 10
        seq_length = 5
        key = jax.random.PRNGKey(42)
        sequences = jax.random.randint(
            key, (n_particles, seq_length), 0, 20, dtype=jnp.int8
        )
        weights = jnp.full(n_particles, 1.0 / n_particles)
        blackjax_state = SMCState(
            particles=sequences,
            weights=weights,
            update_parameters={},
        )
        return SamplerState(
            sequence=sequences,
            key=key,
            blackjax_state=blackjax_state,
            step=jnp.array(0),
            additional_fields={"beta": jnp.array(-1.0)},
        )

    def test_run_smc_loop_runs_and_returns_correct_structure(
        self, mock_initial_state: SamplerState
    ) -> None:
        """Test that the SMC loop runs and returns the expected output structure."""
        num_samples = 5
        n_particles = mock_initial_state.sequence.shape[0]

        def mock_fitness_fn(
            key: PRNGKeyArray,
            sequence: Array,
            beta: Float | None,
        ) -> Array:
            return jnp.sum(sequence, axis=-1, dtype=jnp.float32)

        def mock_mutation_fn(
            key: PRNGKeyArray, sequence: Array, context: None
        ) -> tuple[Array, None]:
            return sequence + 1, None

        def mock_writer_callback(output) -> None:
            pass

        final_state, _ = run_smc_loop(
            num_samples=num_samples,
            algorithm="BaseSMC",
            resampling_approach="multinomial",
            initial_state=mock_initial_state,
            fitness_fn=mock_fitness_fn,
            mutation_fn=mock_mutation_fn,
            writer_callback=mock_writer_callback,
        )

        assert isinstance(final_state, SamplerState)
        chex.assert_shape(
            final_state.sequence, mock_initial_state.sequence.shape
        )
        assert final_state.step == num_samples

    def test_smc_loop_with_annealing(
        self, mock_initial_state: SamplerState
    ) -> None:
        """Test that the loop correctly uses the annealing schedule."""
        num_samples = 3

        def mock_fitness_fn(
            key: PRNGKeyArray,
            sequence: Array,
            beta: Float | None,
        ) -> Array:
            # Beta should be passed and be a scalar float
            chex.assert_type(beta, float)
            return jnp.sum(sequence, axis=-1) * beta

        def mock_mutation_fn(
            key: PRNGKeyArray, sequence: Array, context: None
        ) -> tuple[Array, None]:
            return sequence, None

        def mock_annealing_fn(i: int, _context=None) -> Float:
            return jnp.array(0.1 * (i + 1), dtype=jnp.float32)

        def mock_writer_callback(output) -> None:
            # Check that the beta value is correctly logged
            assert "beta" in output.state.additional_fields
            chex.assert_type(output.state.additional_fields["beta"], float)

        final_state, _ = run_smc_loop(
            num_samples=num_samples,
            algorithm="AnnealedSMC",
            resampling_approach="systematic",
            initial_state=mock_initial_state,
            fitness_fn=mock_fitness_fn,
            mutation_fn=mock_mutation_fn,
            writer_callback=mock_writer_callback,
            annealing_fn=mock_annealing_fn,
        )

        assert "beta" in final_state.additional_fields
        # The final beta should correspond to the last step
        expected_final_beta = mock_annealing_fn(num_samples - 1)
        chex.assert_trees_all_close(
            final_state.additional_fields["beta"], expected_final_beta
        )

    def test_e2e_smc_sampler_improves_fitness(
        self, mock_initial_state: SamplerState
    ) -> None:
        """An end-to-end test to validate that the SMC sampler improves population fitness."""
        num_samples = 20
        n_particles, seq_length = mock_initial_state.sequence.shape

        # A simple fitness function where lower sequence values are better
        def simple_fitness_fn(
            key: PRNGKeyArray, sequence: Array, beta: Float | None
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

        def mock_writer_callback(output) -> None:
            pass

        # Calculate initial mean fitness
        initial_fitness = jax.vmap(simple_fitness_fn, in_axes=(None, 0, None))(
            None, mock_initial_state.sequence, None
        )
        initial_mean_fitness = jnp.mean(initial_fitness)

        # Run the SMC loop
        final_state, _ = run_smc_loop(
            num_samples=num_samples,
            algorithm="BaseSMC",
            resampling_approach="systematic",
            initial_state=mock_initial_state,
            fitness_fn=simple_fitness_fn,
            mutation_fn=mutation_fn,
            writer_callback=mock_writer_callback,
        )

        # Calculate final mean fitness
        final_fitness = jax.vmap(simple_fitness_fn, in_axes=(None, 0, None))(
            None, final_state.sequence, None
        )
        final_mean_fitness = jnp.mean(final_fitness)

        # Assert that the average fitness has improved
        assert final_mean_fitness > initial_mean_fitness
