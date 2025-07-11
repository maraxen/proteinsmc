
import jax
import jax.numpy as jnp
import chex
import pytest

from proteinsmc.utils.mutation import (
  _revert_x_codons_if_mutated,
  diversify_initial_sequences,
  mutate,
  chunked_mutation_step
)
from proteinsmc.utils.translation import translate


@pytest.fixture
def sample_sequences():
  return jnp.array([[0, 1, 2, 3, 0, 1], [0, 0, 0, 1, 1, 1]])


def test_mutate(sample_sequences):
  key = jax.random.PRNGKey(0)
  mutated = mutate(key, sample_sequences, mutation_rate=0.1, n_states=4)
  chex.assert_shape(mutated, sample_sequences.shape)
  chex.assert_equal(jnp.array_equal(mutated, sample_sequences), False)


def test_mutate_zero_rate(sample_sequences):
  key = jax.random.PRNGKey(0)
  mutated = mutate(key, sample_sequences, mutation_rate=0.0, n_states=4)
  chex.assert_trees_all_equal(mutated, sample_sequences)


def test_revert_x_codons():
  template = jnp.array([0, 1, 2])
  mutated = jnp.array([3, 0, 0])

  reverted = _revert_x_codons_if_mutated(
    template_nucleotide_sequences=template, candidate_nucleotide_sequences=mutated
  )
  chex.assert_trees_all_equal(reverted, template)

  mutated_valid = jnp.array([2, 1, 1])
  reverted_valid = _revert_x_codons_if_mutated(
    template_nucleotide_sequences=template, candidate_nucleotide_sequences=mutated_valid
  )
  chex.assert_trees_all_equal(reverted_valid, mutated_valid)


def test_diversify_initial_sequences(sample_sequences):
  key = jax.random.PRNGKey(0)
  diversified = diversify_initial_sequences(
    key,
    template_sequences=sample_sequences,
    mutation_rate=1.0,
    sequence_type="nucleotide",
  )
  chex.assert_shape(diversified, sample_sequences.shape)

  aa_seqs, is_valid = jax.vmap(translate)(diversified)
  chex.assert_equal(jnp.all(is_valid), True)
  
def test_chunked_mutation_step(sample_sequences):
  key = jax.random.PRNGKey(0)
  mutated = chunked_mutation_step(key, sample_sequences, n_states=4, mutation_rate=0.1, chunk_size=2)
  chex.assert_shape(mutated, sample_sequences.shape)
  chex.assert_equal(jnp.array_equal(mutated, sample_sequences), False)
