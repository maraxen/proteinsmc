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