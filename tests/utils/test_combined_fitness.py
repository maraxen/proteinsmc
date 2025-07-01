import jax
import jax.numpy as jnp
import pytest

from src.utils.combined_fitness import (
    calculate_fitness_batch,
    combine_fitness_scores,
)
from src.utils.types import MPNNModel


# Mock MPNNModel for testing purposes
class MockMPNNModel(MPNNModel):
    def __init__(self):
        pass

    def score(self, key, protein_sequence):
        # Simple mock score: sum of amino acid integers
        return jnp.sum(protein_sequence.astype(jnp.float32))


@pytest.fixture
def mock_mpnn_model():
    return MockMPNNModel()


@pytest.fixture
def sample_sequences():
    # Example sequences: 2 protein sequences of length 3
    # (e.g., corresponding to 9 nucleotides)
    return jnp.array(
        [[0, 1, 2], [3, 4, 5]], dtype=jnp.int32
    )  # Example AA sequences


@pytest.fixture
def sample_nucleotide_sequences():
    # Example sequences: 2 nucleotide sequences of length 9
    return jnp.array(
        [
            [0, 0, 0, 1, 1, 1, 2, 2, 2],  # AAA CCC GGG
            [3, 3, 3, 0, 0, 0, 1, 1, 1],  # TTT AAA CCC
        ],
        dtype=jnp.int32,
    )


@pytest.fixture
def mock_fitness_func1():
    # Mock fitness function: returns a constant value
    def _func(key, nuc_seq, aa_seq):
        return jnp.array(10.0, dtype=jnp.float32)

    return _func


@pytest.fixture
def mock_fitness_func2():
    # Mock fitness function: returns a value based on sequence length
    def _func(key, nuc_seq, aa_seq):
        if nuc_seq is not False:
            return jnp.array(len(nuc_seq), dtype=jnp.float32)
        elif aa_seq is not False:
            return jnp.array(len(aa_seq), dtype=jnp.float32)
        return jnp.array(0.0, dtype=jnp.float32)

    return _func


@pytest.fixture
def mock_cai_score():
    # Mock CAI score function
    def _func(nuc_seq, aa_seq):
        # Simplified mock: sum of nucleotide values
        return jnp.sum(nuc_seq.astype(jnp.float32))

    return _func


@pytest.fixture
def mock_mpnn_score():
    # Mock MPNN score function
    def _func(key, aa_seq, mpnn_model_instance):
        return mpnn_model_instance.score(key, aa_seq)

    return _func


def test_combine_fitness_scores_no_weights():
    fitness_components = {"f1": jnp.array([1.0, 2.0]), "f2": jnp.array([3.0, 4.0])}
    combined = combine_fitness_scores(fitness_components)
    assert jnp.array_equal(combined, jnp.array([4.0, 6.0]))


def test_combine_fitness_scores_with_weights():
    fitness_components = {"f1": jnp.array([1.0, 2.0]), "f2": jnp.array([3.0, 4.0])}
    fitness_weights = jnp.array([0.5, 1.5])
    combined = combine_fitness_scores(fitness_components, fitness_weights)
    assert jnp.array_equal(combined, jnp.array([5.0, 7.0]))  # 0.5*1 + 1.5*3 = 5, 0.5*2 + 1.5*4 = 7


def test_calculate_fitness_batch_protein_sequences(
    mock_mpnn_model,
    sample_sequences,
    mock_mpnn_score,
    mock_fitness_func1,
):
    key = jax.random.PRNGKey(0)
    protein_length = 3
    fitness_funcs = (mock_mpnn_score, mock_fitness_func1)
    fitness_flags = (True, True)

    (combined_fitness, fitness_components, valid_translations) = \
        calculate_fitness_batch(
            key,
            sample_sequences,
            fitness_funcs,
            combine_fitness_scores,
            protein_length,
            fitness_flags,
            mpnn_model_instance=mock_mpnn_model,
        )

    # Expected combined fitness: (0+1+2) + 10 = 13, (3+4+5) + 10 = 22
    assert jnp.array_equal(combined_fitness, jnp.array([13.0, 22.0]))
    assert jnp.array_equal(valid_translations, jnp.array([True, True]))
    assert "fitness_0" in fitness_components
    assert "fitness_1" in fitness_components
    assert jnp.array_equal(fitness_components["fitness_0"], jnp.array([3.0, 12.0]))
    assert jnp.array_equal(fitness_components["fitness_1"], jnp.array([10.0, 10.0]))


def test_calculate_fitness_batch_nucleotide_sequences(
    mock_mpnn_model,
    sample_nucleotide_sequences,
    mock_cai_score,
    mock_fitness_func2,
):
    key = jax.random.PRNGKey(0)
    protein_length = 3  # 9 nucleotides for 3 amino acids
    fitness_funcs = (mock_cai_score, mock_fitness_func2)
    fitness_flags = (True, True)

    (combined_fitness, fitness_components, valid_translations) = \
        calculate_fitness_batch(
            key,
            sample_nucleotide_sequences,
            fitness_funcs,
            combine_fitness_scores,
            protein_length,
            fitness_flags,
            mpnn_model_instance=mock_mpnn_model,
        )

    # Expected combined fitness:
    # Seq1 (AAA CCC GGG): CAI (0+0+0+1+1+1+2+2+2=9) + Func2 (len=9) = 18
    # Seq2 (TTT AAA CCC): CAI (3+3+3+0+0+0+1+1+1=12) + Func2 (len=9) = 21
    assert jnp.array_equal(combined_fitness, jnp.array([18.0, 21.0]))
    assert jnp.array_equal(valid_translations, jnp.array([True, True]))
    assert "fitness_0" in fitness_components
    assert "fitness_1" in fitness_components
    assert jnp.array_equal(fitness_components["fitness_0"], jnp.array([9.0, 12.0]))
    assert jnp.array_equal(fitness_components["fitness_1"], jnp.array([9.0, 9.0]))


def test_calculate_fitness_batch_invalid_translation(
    mock_mpnn_model,
    mock_fitness_func1,
):
    key = jax.random.PRNGKey(0)
    protein_length = 1  # Expecting 3 nucleotides
    # Sequence that will result in an 'X' (invalid) translation
    invalid_nuc_seq = jnp.array([[0, 0, 0]], dtype=jnp.int32)  # AAA -> K (valid)
    # Manually create a sequence that would result in an invalid AA (e.g., -1)
    # For this test, we'll simulate an invalid translation by providing a sequence
    # that would lead to -1 in `translate` if it were real.
    # Since our mock `translate` always returns valid, we'll mock the `valid_translation`
    # directly for this test.

    # To properly test invalid translation, we need to mock the `translate` function
    # or provide a sequence that `src.utils.nucleotide.translate` would mark as invalid.
    # Given the current mock setup, let's simulate the outcome of an invalid translation.

    # For now, let's test with a valid sequence and ensure it's marked valid.
    # A more robust test would involve mocking `translate` to return `has_x_residue=True`
    # or directly providing a sequence that `translate` would handle as invalid.

    # Let's create a sequence that, if translated by the real `translate`,
    # would produce an 'X' (COLABDESIGN_X_INT).
    # For example, a codon like 'TAA' (2,0,0) translates to 'X'.
    nuc_seq_with_x = jnp.array([[2, 0, 0]], dtype=jnp.int32)  # TAA
    protein_length_for_x = 1

    fitness_funcs = (mock_fitness_func1,)
    fitness_flags = (True,)

    (combined_fitness, fitness_components, valid_translations) = \
        calculate_fitness_batch(
            key,
            nuc_seq_with_x,
            fitness_funcs,
            combine_fitness_scores,
            protein_length_for_x,
            fitness_flags,
            mpnn_model_instance=mock_mpnn_model,
        )

    # If translation results in 'X', valid_translation should be False and fitness -inf
    # Note: The mock `translate` in `calculate_fitness_single` needs to be accurate
    # or we need to mock `translate` itself.
    # For now, assuming `translate` correctly identifies 'X' and sets valid_translation.
    assert jnp.array_equal(valid_translations, jnp.array([False]))
    assert jnp.isneginf(combined_fitness[0])


def test_calculate_fitness_batch_no_active_functions(
    sample_sequences,
    mock_mpnn_model,
):
    key = jax.random.PRNGKey(0)
    protein_length = 3
    fitness_funcs = ()  # No fitness functions
    fitness_flags = ()

    (combined_fitness, fitness_components, valid_translations) = \
        calculate_fitness_batch(
            key,
            sample_sequences,
            fitness_funcs,
            combine_fitness_scores,
            protein_length,
            fitness_flags,
            mpnn_model_instance=mock_mpnn_model,
        )

    assert jnp.array_equal(combined_fitness, jnp.array([0.0, 0.0]))
    assert jnp.array_equal(valid_translations, jnp.array([True, True]))
    assert not fitness_components  # Should be empty
