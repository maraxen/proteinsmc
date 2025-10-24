import jax.numpy as jnp
from proteinsmc.oed.phase import detect_phase_boundaries


def test_detect_phase_boundaries_simple():
    params = jnp.array([0.0, 0.1, 0.2, 0.3, 0.4])
    metrics = jnp.array([0.0, 0.0, 1.0, 1.0, 1.0])
    boundaries = detect_phase_boundaries(params, metrics, sensitivity_threshold=0.5)
    assert boundaries.size >= 1

