import jax
import jax.numpy as jnp
import pytest

from src.utils.resampling import resampling_kernel


@pytest.mark.parametrize(
    "log_weights, expected_ess, expected_normalized_weights",
    [
        (
            jnp.array([0.0, 0.0, 0.0, 0.0]),
            4.0,
            jnp.array([0.25, 0.25, 0.25, 0.25]),
        ),  # Uniform weights
        (
            jnp.array([jnp.log(0.8), jnp.log(0.1), jnp.log(0.1), jnp.log(0.0)]),
            1.25,
            jnp.array([0.8, 0.1, 0.1, 0.0]),
        ),  # Skewed weights
        (
            jnp.array([-jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf]),
            0.0,
            jnp.array([0.25, 0.25, 0.25, 0.25]),
        ),  # All -inf log_weights
        (
            jnp.array([0.0, -jnp.inf, -jnp.inf, -jnp.inf]),
            1.0,
            jnp.array([1.0, 0.0, 0.0, 0.0]),
        ),  # One valid, rest -inf
        (
            jnp.array([jnp.nan, 0.0, 0.0, 0.0]),
            0.0,
            jnp.array([0.25, 0.25, 0.25, 0.25]),
        ),  # NaN in weights
    ],
)
def test_resampling_kernel(
    log_weights, expected_ess, expected_normalized_weights
):
    key = jax.random.PRNGKey(0)
    n_particles = len(log_weights)
    sequences = jnp.arange(n_particles).reshape(n_particles, 1)  # Dummy sequences

    resampled_sequences, ess, normalized_weights = resampling_kernel(
        key, sequences, log_weights, n_particles
    )

    assert jnp.isclose(ess, expected_ess, atol=1e-6)
    assert jnp.allclose(normalized_weights, expected_normalized_weights, atol=1e-6)

    # Test that resampled_sequences are drawn from original sequences based on weights
    # This is a probabilistic test, so we'll check if the distribution is roughly correct
    # by running multiple times or checking the indices chosen.
    # For deterministic testing, we can check the sum of chosen indices or similar.
    # Given the nature of random.choice, we can't assert exact sequence content
    # without fixing the random key and knowing the exact output of random.choice.
    # Instead, we'll check the shape and type.
    assert resampled_sequences.shape == sequences.shape
    assert resampled_sequences.dtype == sequences.dtype

    # A more robust test for resampling would involve checking the counts of resampled
    # particles over many runs, but for a unit test, checking the properties of ESS
    # and normalized_weights is usually sufficient.


def test_resampling_kernel_sum_weights_near_zero():
    key = jax.random.PRNGKey(0)
    n_particles = 4
    sequences = jnp.arange(n_particles).reshape(n_particles, 1)
    # Weights that sum to a very small number, should trigger uniform distribution
    log_weights = jnp.array([-100.0, -100.0, -100.0, -100.0])

    resampled_sequences, ess, normalized_weights = resampling_kernel(
        key, sequences, log_weights, n_particles
    )

    assert jnp.isclose(ess, 0.0, atol=1e-6)  # ESS should be 0 if weights are problematic
    assert jnp.allclose(
        normalized_weights, jnp.array([0.25, 0.25, 0.25, 0.25]), atol=1e-6
    )


def test_resampling_kernel_large_weights():
    key = jax.random.PRNGKey(0)
    n_particles = 2
    sequences = jnp.arange(n_particles).reshape(n_particles, 1)
    log_weights = jnp.array([1000.0, 1000.0])  # Large weights

    resampled_sequences, ess, normalized_weights = resampling_kernel(
        key, sequences, log_weights, n_particles
    )

    assert jnp.isclose(ess, 2.0, atol=1e-6)
    assert jnp.allclose(normalized_weights, jnp.array([0.5, 0.5]), atol=1e-6)
