import jax.numpy as jnp
import pytest

from src.utils.annealing_schedules import (
    cosine_schedule_py,
    exponential_schedule,
    linear_schedule,
    static_schedule_py,
)


@pytest.mark.parametrize(
    "p, n_steps, beta_max, expected_beta",
    [
        (0, 10, 1.0, 0.0),  # p <= 1, should be 0
        (1, 10, 1.0, 0.0),  # p <= 1, should be 0
        (2, 10, 1.0, 1.0 / 9.0),  # (1.0 * (2-1) / (10-1))
        (5, 10, 1.0, 4.0 / 9.0),  # (1.0 * (5-1) / (10-1))
        (10, 10, 1.0, 1.0),  # p >= n_steps, should be beta_max
        (11, 10, 1.0, 1.0),  # p >= n_steps, should be beta_max
        (2, 5, 2.0, 2.0 / 4.0),  # (2.0 * (2-1) / (5-1))
    ],
)
def test_linear_schedule(p, n_steps, beta_max, expected_beta):
    beta = linear_schedule(
        jnp.array(p, dtype=jnp.int32),
        jnp.array(n_steps, dtype=jnp.int32),
        jnp.array(beta_max, dtype=jnp.float32),
    )
    assert jnp.isclose(beta, expected_beta)


@pytest.mark.parametrize(
    "p, n_steps, beta_max, rate, expected_beta",
    [
        (0, 10, 1.0, 5.0, 0.0),  # p <= 1, should be 0
        (1, 10, 1.0, 5.0, 0.0),  # p <= 1, should be 0
        (10, 10, 1.0, 5.0, 1.0),  # p >= n_steps, should be beta_max
        (11, 10, 1.0, 5.0, 1.0),  # p >= n_steps, should be beta_max
        (
            2,
            10,
            1.0,
            5.0,
            1.0 / (jnp.exp(5.0 * (1.0 / 9.0)) - 1.0) * (jnp.exp(5.0 * (1.0 / 9.0)) - 1),
        ),
        (
            5,
            10,
            1.0,
            5.0,
            1.0 / (jnp.exp(5.0 * (4.0 / 9.0)) - 1.0) * (jnp.exp(5.0 * (4.0 / 9.0)) - 1),
        ),
    ],
)
def test_exponential_schedule(p, n_steps, beta_max, rate, expected_beta):
    beta = exponential_schedule(
        jnp.array(p, dtype=jnp.int32),
        jnp.array(n_steps, dtype=jnp.int32),
        jnp.array(beta_max, dtype=jnp.float32),
        jnp.array(rate, dtype=jnp.float32),
    )
    assert jnp.isclose(beta, expected_beta)


@pytest.mark.parametrize(
    "p, n_steps, beta_max, expected_beta",
    [
        (0, 10, 1.0, 0.0),  # p <= 1, should be 0
        (1, 10, 1.0, 0.0),  # p <= 1, should be 0
        (10, 10, 1.0, 1.0),  # p >= n_steps, should be beta_max
        (11, 10, 1.0, 1.0),  # p >= n_steps, should be beta_max
        (2, 10, 1.0, 0.5 * (1.0 - jnp.cos(jnp.pi * (1.0 / 9.0)))),
        (5, 10, 1.0, 0.5 * (1.0 - jnp.cos(jnp.pi * (4.0 / 9.0)))),
    ],
)
def test_cosine_schedule(p, n_steps, beta_max, expected_beta):
    beta = cosine_schedule_py(
        jnp.array(p, dtype=jnp.int32),
        jnp.array(n_steps, dtype=jnp.int32),
        jnp.array(beta_max, dtype=jnp.float32),
    )
    assert jnp.isclose(beta, expected_beta)


@pytest.mark.parametrize(
    "beta_max, expected_beta",
    [
        (1.0, 1.0),
        (0.5, 0.5),
        (0.0, 0.0),
    ],
)
def test_static_schedule(beta_max, expected_beta):
    beta = static_schedule_py(jnp.array(beta_max, dtype=jnp.float32))
    assert jnp.isclose(beta, expected_beta)
