
import jax
import jax.numpy as jnp
import chex
import pytest

from proteinsmc.scoring.mpnn import make_mpnn_score


class MockMPNNModel:
  """A mock MPNNModel class for testing purposes."""

  def score(self, seq_numeric, key):
    return jnp.sum(seq_numeric)


@pytest.fixture
def mock_model():
  """Fixture for the mock MPNN model."""
  return MockMPNNModel()


def test_mpnn_score(mock_model):
  """Test the mpnn_score function with a mock model."""
  key = jax.random.PRNGKey(0)
  protein_sequence = jnp.array([1, 2, 3])

  score_func = make_mpnn_score(mock_model)

  expected_score = jnp.sum(protein_sequence)
  chex.assert_trees_all_equal(score_func(key, protein_sequence), expected_score)


def test_mpnn_score_empty_sequence(mock_model):
  """Test the mpnn_score function with an empty sequence."""
  key = jax.random.PRNGKey(0)
  protein_sequence = jnp.array([])

  score_func = make_mpnn_score(mock_model)

  expected_score = 0.0
  chex.assert_trees_all_equal(score_func(key, protein_sequence), expected_score)
