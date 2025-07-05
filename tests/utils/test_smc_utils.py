import jax
import jax.numpy as jnp
import pytest

from src.utils.constants import CODON_INT_TO_RES_INT_JAX, COLABDESIGN_X_INT
from src.utils.smc_utils import _revert_x_codons_if_mutated, initial_mutation_kernel_no_x_jax


@pytest.fixture
def protein_length():
  return 3  # Corresponds to 9 nucleotides


@pytest.fixture
def n_nuc_alphabet_size():
  return 4  # A, C, G, T


@pytest.fixture
def particles_template_population():
  # Population of 2 particles, each 9 nucleotides long
  # Particle 1: AAA CCC GGG (K P G)
  # Particle 2: TTT AAA CCC (F K P)
  return jnp.array(
    [
      [0, 0, 0, 1, 1, 1, 2, 2, 2],
      [3, 3, 3, 0, 0, 0, 1, 1, 1],
    ],
    dtype=jnp.int32,
  )


def test_initial_mutation_kernel_no_x_jax_no_mutation(
  protein_length, n_nuc_alphabet_size, particles_template_population
):
  key = jax.random.PRNGKey(0)
  mu_nuc = 0.0  # No mutation

  mutated_particles = initial_mutation_kernel_no_x_jax(
    key,
    particles_template_population,
    mu_nuc,
    n_nuc_alphabet_size,
    protein_length,
  )

  assert jnp.array_equal(mutated_particles, particles_template_population)


def test_initial_mutation_kernel_no_x_jax_all_mutations_no_x(
  protein_length, n_nuc_alphabet_size, particles_template_population
):
  key = jax.random.PRNGKey(0)
  mu_nuc = 1.0  # All nucleotides attempt to mutate

  mutated_particles = initial_mutation_kernel_no_x_jax(
    key,
    particles_template_population,
    mu_nuc,
    n_nuc_alphabet_size,
    protein_length,
  )

  # Verify that no 'X' codons were introduced
  for i in range(mutated_particles.shape[0]):
    for j in range(protein_length):
      codon = mutated_particles[i, j * 3 : j * 3 + 3]
      n1, n2, n3 = codon[0], codon[1], codon[2]
      translated_aa = CODON_INT_TO_RES_INT_JAX[n1, n2, n3]
      assert translated_aa != COLABDESIGN_X_INT


def test_check_and_revert_x_codons_single_particle(protein_length):
  # Template: AAA CCC GGG (K P G)
  template_nucs = jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=jnp.int32)

  # Proposed: AAA TAA GGG (K X G) - TAA is (3,0,0)
  proposed_nucs_with_x = jnp.array([0, 0, 0, 3, 0, 0, 2, 2, 2], dtype=jnp.int32)

  # Expected: AAA CCC GGG (K P G) - CCC (1,1,1) should be reverted
  expected_reverted_nucs = jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=jnp.int32)

  reverted_nucs = _revert_x_codons_if_mutated(template_nucs, proposed_nucs_with_x, protein_length)

  assert jnp.array_equal(reverted_nucs, expected_reverted_nucs)


def test_initial_mutation_kernel_no_x_jax_reverts_x_codon(
  protein_length, n_nuc_alphabet_size, particles_template_population
):
  key = jax.random.PRNGKey(0)
  mu_nuc = 1.0  # Force mutation

  mutated_particles = initial_mutation_kernel_no_x_jax(
    key,
    particles_template_population,
    mu_nuc,
    n_nuc_alphabet_size,
    protein_length,
  )

  # Verify that no 'X' codons were introduced after reversion
  for i in range(mutated_particles.shape[0]):
    for j in range(protein_length):
      codon = mutated_particles[i, j * 3 : j * 3 + 3]
      n1, n2, n3 = codon[0], codon[1], codon[2]
      translated_aa = CODON_INT_TO_RES_INT_JAX[n1, n2, n3]
      assert translated_aa != COLABDESIGN_X_INT


def test_initial_mutation_kernel_no_x_jax_partial_mutation(
  protein_length, n_nuc_alphabet_size, particles_template_population
):
  key = jax.random.PRNGKey(1)  # Use a different key
  mu_nuc = 0.1  # Small mutation rate

  mutated_particles = initial_mutation_kernel_no_x_jax(
    key,
    particles_template_population,
    mu_nuc,
    n_nuc_alphabet_size,
    protein_length,
  )

  # Ensure some mutations happened (unlikely to be exactly 0 with mu_nuc > 0)
  assert not jnp.array_equal(mutated_particles, particles_template_population)

  # Verify no 'X' codons were introduced
  for i in range(mutated_particles.shape[0]):
    for j in range(protein_length):
      codon = mutated_particles[i, j * 3 : j * 3 + 3]
      n1, n2, n3 = codon[0], codon[1], codon[2]
      translated_aa = CODON_INT_TO_RES_INT_JAX[n1, n2, n3]
      assert translated_aa != COLABDESIGN_X_INT
