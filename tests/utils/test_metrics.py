import jax.numpy as jnp
import pytest

from src.utils.metrics import calculate_position_entropy, shannon_entropy


@pytest.mark.parametrize(
    "pos_seqs, expected_entropy",
    [
        (jnp.array([0, 0, 0]), 0.0),  # All same
        (jnp.array([0, 1, 0]), 0.636514168),  # Two 0s, one 1
        (jnp.array([0, 1, 2]), 1.0986123),  # All unique
        (jnp.array([]), 0.0),  # Empty sequence
    ],
)
def test_calculate_position_entropy(pos_seqs, expected_entropy):
    if pos_seqs.size == 0:
        entropy = calculate_position_entropy(pos_seqs)
        assert jnp.isclose(entropy, expected_entropy)
    else:
        entropy = calculate_position_entropy(pos_seqs)
        assert jnp.isclose(entropy, expected_entropy, atol=1e-6)


@pytest.mark.parametrize(
    "seqs, expected_entropy",
    [
        (jnp.array([[0, 0, 0], [0, 0, 0]]), 0.0),  # All same sequences
        (jnp.array([[0, 1, 2], [0, 1, 2]]), 0.0),  # All same sequences
        (jnp.array([[0, 0, 0], [1, 1, 1]]), 0.69314718),  # Two distinct sequences
        (
            jnp.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
            1.0986123,  # All unique at each position
        ),
        (jnp.array([]).reshape(0, 3), 0.0),  # Empty batch of sequences
        (jnp.array([[0, 0, 0]]), 0.0),  # Single sequence, all same
        (jnp.array([[0, 1, 2]]), 1.0986123),  # Single sequence, all unique
    ],
)
def test_shannon_entropy(seqs, expected_entropy):
    entropy = shannon_entropy(seqs)
    assert jnp.isclose(entropy, expected_entropy, atol=1e-6)


def test_shannon_entropy_empty_input():
    # Test with an empty array, but with a defined shape for the second dimension
    empty_seqs = jnp.empty((0, 5), dtype=jnp.int32)
    entropy = shannon_entropy(empty_seqs)
    assert jnp.isclose(entropy, 0.0)


def test_shannon_entropy_single_position_sequences():
    seqs = jnp.array([[0], [1], [0]])
    # Expected: position entropy for [0,1,0] is 0.636514168
    # Since there's only one position, total entropy is 0.636514168 / 1 = 0.636514168
    expected_entropy = 0.636514168
    entropy = shannon_entropy(seqs)
    assert jnp.isclose(entropy, expected_entropy, atol=1e-6)
