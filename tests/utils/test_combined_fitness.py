import jax
import jax.numpy as jnp
import pytest

from src.scoring.cai import cai_score
from src.scoring.mpnn import mpnn_score
from src.utils.combined_fitness import (
  calculate_fitness_population,
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
  return jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)  # Example AA sequences


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


def test_combine_fitness_scores_no_weights():
  fitness_components = {"f1": jnp.array([1.0, 2.0]), "f2": jnp.array([3.0, 4.0])}
  combined = combine_fitness_scores(fitness_components)
  assert jnp.array_equal(combined, jnp.array([4.0, 6.0]))


def test_combine_fitness_scores_with_weights():
  fitness_components = {"f1": jnp.array([1.0, 2.0]), "f2": jnp.array([3.0, 4.0])}
  fitness_weights = jnp.array([0.5, 1.5])
  combined = combine_fitness_scores(fitness_components, fitness_weights)
  assert jnp.array_equal(combined, jnp.array([5.0, 7.0]))  # 0.5*1 + 1.5*3 = 5, 0.5*2 + 1.5*4 = 7


def test_calculate_fitness_population_protein_sequences(
  mock_mpnn_model,
  sample_sequences,
):
  key = jax.random.PRNGKey(0)
  protein_length = 3
  fitness_funcs = (mpnn_score,)
  fitness_flags = (True,)

  (combined_fitness, fitness_components, valid_translations) = calculate_fitness_population(
    key,
    sample_sequences,
    fitness_funcs,
    combine_fitness_scores,
    protein_length,
    fitness_flags,
    mpnn_model_instance=mock_mpnn_model,
  )

  # Expected combined fitness: (0+1+2) = 3, (3+4+5) = 12
  assert jnp.allclose(combined_fitness, jnp.array([3.0, 12.0]), atol=1e-3)
  assert jnp.array_equal(valid_translations, jnp.array([True, True]))
  assert "fitness_0" in fitness_components
  assert jnp.allclose(fitness_components["fitness_0"], jnp.array([3.0, 12.0]), atol=1e-3)


def test_calculate_fitness_population_nucleotide_sequences(
  sample_nucleotide_sequences,
):
  key = jax.random.PRNGKey(0)
  protein_length = 3  # 9 nucleotides for 3 amino acids
  fitness_funcs = (cai_score,)
  fitness_flags = (True,)

  (combined_fitness, fitness_components, valid_translations) = calculate_fitness_population(
    key,
    sample_nucleotide_sequences,
    fitness_funcs,
    combine_fitness_scores,
    protein_length,
    fitness_flags,
    mpnn_model_is_active=False,
  )

  # Expected combined fitness:
  # Seq1 (AAA CCC GGG): CAI(...) = 0.4158
  # Seq2 (TTT AAA CCC): CAI(...) = 0.62118765
  assert jnp.allclose(combined_fitness, jnp.array([0.4158, 0.62118765]), atol=1e-3)
  assert jnp.array_equal(valid_translations, jnp.array([True, True]))
  assert "fitness_0" in fitness_components
  assert jnp.allclose(fitness_components["fitness_0"], jnp.array([0.345, 0.414]), atol=1e-3)
