
import jax.numpy as jnp
import chex
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
def test_calculate_logZ_increment(
  log_weights: jnp.ndarray, population_size: int, expected: float
) -> None:
  """Test calculation of logZ increment under various conditions."""
  result = calculate_logZ_increment(log_weights, population_size)
  chex.assert_trees_all_close(result, expected)


@pytest.mark.parametrize(
  "pos_seqs, expected",
  [
    (jnp.array([0, 0, 1, 1]), -2 * (0.5 * jnp.log(0.5))),
    (jnp.array([0, 1, 2, 3]), -4 * (0.25 * jnp.log(0.25))),
    (jnp.array([0, 0, 0, 0]), 0.0),
    (jnp.array([]), 0.0),
  ],
)
def test_calculate_position_entropy(
  pos_seqs: jnp.ndarray, expected: float
) -> None:
  """Test calculation of entropy for a single position."""
  if pos_seqs.size == 0:
    assert expected == 0.0
    return
  result = calculate_position_entropy(pos_seqs)
  chex.assert_trees_all_close(result, expected)


@pytest.mark.parametrize(
  "seqs, expected",
  [
    (jnp.array([[0, 0], [0, 0]]), 0.0),
    (jnp.array([[0, 1], [0, 1]]), 0.0),
    (jnp.array([]), 0.0),
  ],
)
def test_shannon_entropy(
  seqs: jnp.ndarray, expected: float
) -> None:
  """Test Shannon entropy calculation for a set of sequences."""
  if seqs.size == 0:
    assert expected == 0.0
    return
  result = shannon_entropy(seqs)
  chex.assert_trees_all_close(result, expected)
