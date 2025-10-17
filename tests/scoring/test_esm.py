"""Tests for ESM scoring functions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from proteinsmc.scoring.esm import make_esm_score
from proteinsmc.utils.esm import remap_sequences


# Define a mock model that simulates the behavior of the real ESM model
class MockESMOutput(eqx.Module):
  """Mock ESM model output."""

  logits: jnp.ndarray


class MockESMModel(eqx.Module):
  """Mock ESM model."""

  def __call__(self, sequence: jnp.ndarray) -> MockESMOutput:
    """Mock model forward pass."""
    batch, seq_len = sequence.shape
    vocab_size = 64  # Standard ESM vocab size
    # Use a fixed key for deterministic output in the mock
    key = jax.random.PRNGKey(0)
    logits = jax.random.normal(key, (batch, seq_len, vocab_size))
    return MockESMOutput(logits=logits)


@pytest.fixture
def mock_load_model(monkeypatch: pytest.MonkeyPatch) -> None:
  """Mocks the load_model function to avoid network calls."""
  mock_model = MockESMModel()
  # The original load_model takes model_name and a JAX key
  monkeypatch.setattr(
    "proteinsmc.scoring.esm.load_model",
    lambda model_name, key: mock_model,
  )


def test_make_esm_score_factory(mock_load_model: None) -> None:
  """Tests that the factory function returns a valid, callable scoring function."""
  score_fn = make_esm_score(model_name="esmc_300m")
  assert callable(score_fn), "The factory did not return a callable function."


def test_esm_score_execution(mock_load_model: None) -> None:
  """Tests that the created scoring function executes without errors."""
  score_fn = make_esm_score(model_name="esmc_300m")
  key = jax.random.PRNGKey(42)
  sequence = jax.random.randint(key, (10,), 0, 20, dtype=jnp.int8)

  score = score_fn(sequence, key)

  assert isinstance(score, jnp.ndarray)
  assert score.shape == ()
  assert score.dtype == jnp.float32

