"""Tests for the SMC sampler."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import chex
import jax
import jax.numpy as jnp
import pytest
from blackjax.smc.base import SMCInfo, SMCState

from proteinsmc.models.sampler_base import SamplerOutput, SamplerState
from proteinsmc.sampling.particle_systems.smc import (
    UpdateInfo,
    create_smc_loop_func,
    resample,
    run_smc_loop,
)

if TYPE_CHECKING:
    from jaxtyping import Array, Float, Int, PRNGKeyArray


class TestResample:
    """Test the resample function."""

    @pytest.mark.parametrize(
        "resampling_approach",
        ["systematic", "multinomial", "stratified", "residual"],
    )
    def test_resample_output_shape_and_type(
        self, resampling_approach: str
    ) -> None:
        """Test that resample returns the correct shape and type."""
        key = jax.random.PRNGKey(42)
        weights = jnp.array([0.1, 0.2, 0.3, 0.4])
        num_samples = 10
        ancestors = resample(resampling_approach, key, weights, num_samples)
        chex.assert_shape(ancestors, (num_samples,))
        chex.assert_type(ancestors, jnp.int32)

    def test_resample_invalid_approach(self) -> None:
        """Test that resample raises an error for an invalid approach."""
        key = jax.random.PRNGKey(42)
        weights = jnp.array([0.1, 0.2, 0.3, 0.4])
        num_samples = 10
        with pytest.raises(ValueError, match="Unknown resampling approach"):
            resample("invalid_approach", key, weights, num_samples)


class TestCreateSMCLoopFunc:
    """Test the create_smc_loop_func function."""

    def test_create_smc_loop_func_base_smc(self) -> None:
        """Test that create_smc_loop_func returns a callable for BaseSMC."""

        def mutation_fn(
            key: PRNGKeyArray, state: SamplerState
        ) -> tuple[SamplerState, UpdateInfo]:
            return state, UpdateInfo()

        smc_loop_func = create_smc_loop_func("BaseSMC", "systematic", mutation_fn)
        assert callable(smc_loop_func)

    def test_create_smc_loop_func_not_implemented(self) -> None:
        """Test that create_smc_loop_func raises an error for an unsupported algorithm."""
        with pytest.raises(
            NotImplementedError,
            match="Adaptive Tempered SMC algorithm is not implemented in the loop function.",
        ):
            create_smc_loop_func(
                "AdaptiveTemperedSMC", "systematic", lambda x, y, z: (x, None)
            )


@pytest.fixture
def mock_initial_state() -> SamplerState:
    """Create a mock initial state for the SMC loop."""
    population_size = 4
    seq_length = 3
    particles = jax.random.randint(
        jax.random.PRNGKey(0),
        (population_size, seq_length),
        0,
        20,
        dtype=jnp.int8,
    )
    weights = jnp.ones(population_size) / population_size
    blackjax_state = SMCState(
        particles=particles,
        weights=weights,
        update_parameters=None,
    )
    return SamplerState(
        sequence=particles,
        key=jax.random.PRNGKey(42),
        blackjax_state=blackjax_state,
        step=jnp.array(0, dtype=jnp.int32),
        additional_fields={"beta": jnp.array(-1.0, dtype=jnp.float32)},
    )


class TestRunSMCLoop:
    """Test the run_smc_loop function."""

    def test_run_smc_loop_basic(self, mock_initial_state: SamplerState) -> None:
        """Test basic run_smc_loop functionality."""
        num_samples = 2
        algorithm = "BaseSMC"
        resampling_approach = "multinomial"

        def mock_fitness_fn(
            key: PRNGKeyArray, sequence: Array, beta: Float | None
        ) -> Array:
            return jnp.array(1.0, dtype=jnp.float32)

        def mock_mutation_fn(
            key: PRNGKeyArray, sequence: Array, context: None
        ) -> tuple[Array, UpdateInfo]:
            return sequence, UpdateInfo()

        def mock_writer_callback(data: dict) -> None:
            pass

        final_state = run_smc_loop(
            num_samples=num_samples,
            algorithm=algorithm,
            resampling_approach=resampling_approach,
            initial_state=mock_initial_state,
            fitness_fn=mock_fitness_fn,
            mutation_fn=mock_mutation_fn,
            writer_callback=mock_writer_callback,
        )
        assert isinstance(final_state, SamplerState)
        assert final_state.step == num_samples
        chex.assert_shape(
            final_state.sequence,
            (
                mock_initial_state.sequence.shape[0],
                mock_initial_state.sequence.shape[1],
            ),
        )

    def test_run_smc_loop_end_to_end(self, mock_initial_state: SamplerState) -> None:
        """Test the run_smc_loop with a simple end-to-end case."""
        num_samples = 5
        algorithm = "AnnealedSMC"
        resampling_approach = "systematic"

        def fitness_fn(key: PRNGKeyArray, sequence: Array, beta: Float) -> Array:
            # A simple fitness function that prefers sequences with higher sums
            return jnp.sum(sequence) * beta

        def mutation_fn(
            key: PRNGKeyArray, sequence: Array, context: None
        ) -> tuple[Array, UpdateInfo]:
            # A simple mutation that adds 1 to a random position
            pos = jax.random.randint(key, (), 0, sequence.shape[0])
            return sequence.at[pos].set(sequence[pos] + 1), UpdateInfo()

        def annealing_fn(t: Int, _context: None) -> Float:
            return t / num_samples

        def writer_callback(output: SamplerOutput) -> None:
            # Check that the output is of the correct type
            assert isinstance(output, SamplerOutput)
            assert hasattr(output, "sequences")
            assert hasattr(output, "fitness")

        final_state = run_smc_loop(
            num_samples=num_samples,
            algorithm=algorithm,
            resampling_approach=resampling_approach,
            initial_state=mock_initial_state,
            fitness_fn=fitness_fn,
            mutation_fn=mutation_fn,
            writer_callback=writer_callback,
            annealing_fn=annealing_fn,
        )

        assert isinstance(final_state, SamplerState)
        assert final_state.step == num_samples

        # Check that the fitness of the particles has generally increased
        initial_fitness = jnp.mean(
            jax.vmap(fitness_fn, in_axes=(None, 0, None))(
                jax.random.PRNGKey(0), mock_initial_state.sequence, 1.0
            )
        )
        final_fitness = jnp.mean(
            jax.vmap(fitness_fn, in_axes=(None, 0, None))(
                jax.random.PRNGKey(1), final_state.sequence, 1.0
            )
        )
        assert final_fitness > initial_fitness
