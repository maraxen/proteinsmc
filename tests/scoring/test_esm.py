import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from proteinsmc.scoring.esm import make_esm_score

# Define a mock model that simulates the behavior of the real ESM model
class MockESMOutput(eqx.Module):
    logits: jnp.ndarray

class MockESMModel(eqx.Module):
    def __call__(self, sequence):
        batch, seq_len = sequence.shape
        vocab_size = 64  # Standard ESM vocab size
        # Use a fixed key for deterministic output in the mock
        key = jax.random.PRNGKey(0)
        logits = jax.random.normal(key, (batch, seq_len, vocab_size))
        return MockESMOutput(logits=logits)

@pytest.fixture
def mock_load_model(monkeypatch):
    """Mocks the load_model function to avoid network calls."""
    mock_model = MockESMModel()
    # The original load_model takes model_name and a JAX key
    monkeypatch.setattr("proteinsmc.scoring.esm.load_model", lambda model_name, key: mock_model)

def test_make_esm_score_factory(mock_load_model):
    """Tests that the factory function returns a valid, callable scoring function."""
    score_fn = make_esm_score(model_name="esmc_300m")
    assert callable(score_fn), "The factory did not return a callable function."

def test_esm_score_execution(mock_load_model):
    """Tests that the created scoring function executes without errors."""
    score_fn = make_esm_score(model_name="esmc_300m")
    key = jax.random.PRNGKey(42)
    sequence = jax.random.randint(key, (10,), 0, 20, dtype=jnp.int8)

    score = score_fn(key, sequence)

    assert isinstance(score, jnp.ndarray)
    assert score.shape == ()
    assert score.dtype == jnp.float32
=======
from unittest.mock import patch, MagicMock

from proteinsmc.scoring.esm import make_esm_score
from proteinsmc.utils.esm import remap_sequences

@patch("proteinsmc.scoring.esm.load_model")
def test_make_esm_score_factory(mock_load_model):
    """Tests that the factory function returns a valid, callable scoring function."""
    # Arrange
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model

    # Act
    score_fn = make_esm_score(model_name="esmc_300m")

    # Assert
    assert callable(score_fn)
    mock_load_model.assert_called_once()

@patch("proteinsmc.scoring.esm.load_model")
def test_esm_score_calculation(mock_load_model):
    """
    Tests the logic of the scoring function by comparing its output
    to a manual calculation.
    """
    # Arrange
    key = jax.random.PRNGKey(42)
    seq_len = 10
    vocab_size = 33  # ESM vocab size, includes special tokens

    # Define a known sequence
    sequence = jnp.arange(seq_len, dtype=jnp.int8)

    # Manually remap sequence to what the score function will use internally
    remapped_sequence = remap_sequences(sequence)
    remapped_sequence_with_batch = remapped_sequence[None, :]

    # Create mock logits that the model will return
    mock_logits = jax.random.normal(key, (1, seq_len + 2, vocab_size))

    # Set up the mock model
    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.logits = mock_logits
    mock_model.return_value = mock_output
    mock_load_model.return_value = mock_model

    # Create the scoring function
    score_fn = make_esm_score(model_name="esmc_300m", seed=42)

    # Act
    # Call the scoring function with the original sequence
    actual_score = score_fn(key, sequence)

    # Manually calculate the expected score based on the mocked logits
    log_probs = jax.nn.log_softmax(mock_logits, axis=-1)
    seq_indices = remapped_sequence_with_batch[..., None]
    seq_log_probs = jnp.take_along_axis(log_probs, seq_indices, axis=-1).squeeze(-1)
    expected_score = jnp.sum(seq_log_probs) / (remapped_sequence_with_batch.shape[1] + 1e-8)

    # Assert
    assert isinstance(actual_score, jnp.ndarray)
    assert actual_score.shape == ()
    assert jnp.allclose(actual_score, expected_score.astype(jnp.float32))
