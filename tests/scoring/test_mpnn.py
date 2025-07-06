import jax
import jax.numpy as jnp
import pytest

from proteinsmc.scoring.mpnn import mpnn_score


class MockMPNNModel:
  """A mock MPNNModel class for testing purposes."""

  def score(self, seq_numeric, key):
    # Simple mock scoring: return the sum of the amino acid integers
    return jnp.sum(seq_numeric)


@pytest.fixture
def mock_model():
  """Fixture for the mock MPNN model."""
  return MockMPNNModel()


def test_mpnn_score(mock_model):
  """Test the mpnn_score function with a mock model."""
  key = jax.random.PRNGKey(0)
  # Represents a simple protein sequence, e.g., 'ACD'
  protein_sequence = jnp.array([1, 2, 3])

  score = mpnn_score(key, protein_sequence, mock_model)

  # The mock score is the sum of the sequence integers
  expected_score = jnp.sum(protein_sequence)
  assert score == expected_score


def test_mpnn_score_empty_sequence(mock_model):
  """Test the mpnn_score function with an empty sequence."""
  key = jax.random.PRNGKey(0)
  protein_sequence = jnp.array([])

  score = mpnn_score(key, protein_sequence, mock_model)

  expected_score = 0.0
  assert score == expected_score
