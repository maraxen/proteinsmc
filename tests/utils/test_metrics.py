import jax.numpy as jnp
import pytest

from proteinsmc.utils.metrics import (
  calculate_logZ_increment,
  calculate_position_entropy,
  shannon_entropy,
)


@pytest.mark.parametrize(
  "log_weights, population_size, expected",
  [
    (jnp.array([0.0, 0.0, 0.0]), 3, 0.0),
    (jnp.log(jnp.array([1.0, 2.0, 3.0])), 3, jnp.log(2.0)),
    (jnp.array([-jnp.inf, -jnp.inf]), 2, -jnp.inf),
    (
      jnp.array([1.0, 2.0, -jnp.inf]),
      3,
      jnp.log(jnp.exp(1.0) + jnp.exp(2.0)) - jnp.log(3.0),
    ),
    (jnp.array([]), 0, -jnp.inf),
    (jnp.array([]), 1, -jnp.inf),
  ],
)
def test_calculate_logZ_increment(log_weights, population_size, expected):
  """Test calculation of logZ increment under various conditions."""
  result = calculate_logZ_increment(log_weights, population_size)
  assert jnp.allclose(result, expected, equal_nan=True)


@pytest.mark.parametrize(
  "pos_seqs, expected",
  [
    (jnp.array([0, 0, 1, 1]), -2 * (0.5 * jnp.log(0.5))),
    (jnp.array([0, 1, 2, 3]), -4 * (0.25 * jnp.log(0.25))),
    (jnp.array([0, 0, 0, 0]), 0.0),
  ],
)
def test_calculate_position_entropy(pos_seqs, expected):
  """Test calculation of entropy for a single position."""
  result = calculate_position_entropy(pos_seqs)
  assert jnp.allclose(result, expected)


@pytest.mark.parametrize(
  "seqs, expected",
  [
    (jnp.array([[0, 0], [0, 0]]), 0.0),
    (jnp.array([[0, 1], [0, 1]]), 0.0),
    (
      jnp.array([[0, 1], [1, 0]]),
      (-2 * (0.5 * jnp.log(0.5))),
    ),
    (jnp.array([]), 0.0),
  ],
)
def test_shannon_entropy(seqs, expected):
  """Test Shannon entropy calculation for a set of sequences."""
  result = shannon_entropy(seqs)
  assert jnp.allclose(result, expected)
