import jax
import jax.numpy as jnp
import pytest

from src.utils.mutate import (
  _revert_x_codons_if_mutated,
  diversify_initial_sequences,
  mutate,
)
from src.utils.translate import translate


@pytest.fixture
def sample_sequences():
  return jnp.array([[0, 1, 2, 3, 0, 1], [0, 0, 0, 1, 1, 1]])


def test_mutate(sample_sequences):
  key = jax.random.PRNGKey(0)
  mutated = mutate(key, sample_sequences, mutation_rate=0.1, n_states=4)
  assert mutated.shape == sample_sequences.shape
  assert not jnp.array_equal(mutated, sample_sequences)


def test_mutate_zero_rate(sample_sequences):
  key = jax.random.PRNGKey(0)
  mutated = mutate(key, sample_sequences, mutation_rate=0.0, n_states=4)
  assert jnp.array_equal(mutated, sample_sequences)


def test_revert_x_codons():
  template = jnp.array([0, 1, 2])
  mutated = jnp.array([3, 0, 0])

  reverted = _revert_x_codons_if_mutated(
    template_nucleotide_sequences=template,
    candidate_nucleotide_sequences=mutated,
    sequence_length=1,
  )
  assert jnp.array_equal(reverted, template)

  mutated_valid = jnp.array([2, 1, 1])
  reverted_valid = _revert_x_codons_if_mutated(
    template_nucleotide_sequences=template,
    candidate_nucleotide_sequences=mutated_valid,
    sequence_length=1,
  )
  assert jnp.array_equal(reverted_valid, mutated_valid)


def test_diversify_initial_sequences(sample_sequences):
  key = jax.random.PRNGKey(0)
  diversified = diversify_initial_sequences(
    key,
    template_sequences=sample_sequences,
    mutation_rate=1.0,
    n_states=4,
    sequence_length=2,
  )
  assert diversified.shape == sample_sequences.shape

  aa_seqs, has_x_flags = jax.vmap(translate)(diversified)
  assert not jnp.any(has_x_flags)
