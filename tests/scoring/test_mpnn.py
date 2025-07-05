import jax
import jax.numpy as jnp
import pytest

from src.scoring.mpnn import mpnn_score
from src.utils.types import MPNNModel, ProteinSequence


class MockMPNNModel(MPNNModel):
    def __init__(self):
        pass

    def score(self, protein_sequence: ProteinSequence, key: jax.random.PRNGKey):
        return jnp.sum(protein_sequence.astype(jnp.float32))


@pytest.fixture
def mock_mpnn_model():
    return MockMPNNModel()


@pytest.fixture
def aa_seq() -> ProteinSequence:
    return jnp.array([11, 13, 7], dtype=jnp.int32)


def test_mpnn_score(mock_mpnn_model, aa_seq):
    key = jax.random.PRNGKey(0)
    score = mpnn_score(key, aa_seq, mock_mpnn_model)
    assert jnp.isclose(score, 31.0)