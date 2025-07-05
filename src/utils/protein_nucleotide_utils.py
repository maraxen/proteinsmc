from typing import Callable

import jax
import jax.numpy as jnp
from colabdesign.mpnn.model import mk_mpnn_model
from jax import random, vmap

from src.scoring.cai import cai_score
from src.scoring.mpnn import mpnn_score

from .constants import NUCLEOTIDES_CHAR, NUCLEOTIDES_INT_MAP, RES_TO_CODON_CHAR
from .nucleotide import translate
from .smc_utils import (
  diversify_initial_sequences,
)


def calculate_population_fitness(
  key: jax.Array,
  population: jax.Array,
  mpnn_model_instance: mk_mpnn_model,
  mpnn_model_is_active: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  vmapped_translate = vmap(translate, in_axes=(0,))
  aa_seqs, has_x_flags = vmapped_translate(population)

  vmapped_cai_score = vmap(cai_score, in_axes=(0, 0))
  cai_values = vmapped_cai_score(population, aa_seqs)

  vmapped_mpnn_score = vmap(mpnn_score, in_axes=(0, 0, None))
  keys_for_mpnn = jax.random.split(key, population.shape[0])
  if mpnn_model_is_active:
    mpnn_scores = vmapped_mpnn_score(keys_for_mpnn, aa_seqs, mpnn_model_instance)
  else:
    mpnn_scores = jnp.zeros_like(cai_values)

  combined_fitness = cai_values + mpnn_scores

  return combined_fitness, cai_values, mpnn_scores, aa_seqs, has_x_flags


def get_protein_nucleotide_initial_population_fn(
  initial_aa_seq_char: str, protein_length: int, initial_population_mutation_rate: float
) -> Callable[[jax.Array, int], jax.Array]:
  N_nuc_total = 3 * protein_length
  initial_nucleotide_sequences = [NUCLEOTIDES_INT_MAP["A"]] * N_nuc_total
  try:
    for i in range(protein_length):
      aa_char = initial_aa_seq_char[i]
      codon_char_list = list(RES_TO_CODON_CHAR[aa_char])
      for j in range(3):
        initial_nucleotide_sequences[3 * i + j] = NUCLEOTIDES_INT_MAP[codon_char_list[j]]
  except KeyError as e:
    raise ValueError(
      f"Failed to generate initial JAX nucleotide template from AA '{e}'. "
      f"Check RES_TO_CODON_CHAR and initial_aa_seq_char."
    ) from e
  initial_nucleotide_template_one_seq_jax = jnp.array(initial_nucleotide_sequences, dtype=jnp.int32)

  def initial_population_fn(key: jax.Array, n_particles: int) -> jax.Array:
    _particles_jax_template_population = jnp.tile(
      initial_nucleotide_template_one_seq_jax, (n_particles, 1)
    )
    particles_jax_initial = diversify_initial_sequences(
      key=key,
      template_sequences=_particles_jax_template_population,
      mutation_rate=initial_population_mutation_rate,
      n_states=len(NUCLEOTIDES_CHAR),
      sequence_length=protein_length,
      nucleotide=True,
    )
    return particles_jax_initial

  return initial_population_fn


def get_protein_nucleotide_fitness_fns(
  mpnn_model_instance: mk_mpnn_model, mpnn_model_is_active_static: bool, protein_length: int
) -> tuple[Callable[[jax.Array], jax.Array], Callable[[jax.Array], jax.Array]]:
  def population_fitness_fn(particles_population: jax.Array) -> jax.Array:
    key_for_fitness, _ = random.split(random.PRNGKey(0))
    fitness_values, _, _, _, _ = calculate_population_fitness(
      key_for_fitness,
      particles_population,
      mpnn_model_instance,
      mpnn_model_is_active_static,
    )
    return fitness_values

  def single_fitness_fn(particle: jax.Array) -> jax.Array:
    key_for_fitness, _ = random.split(random.PRNGKey(0))  # Dummy key
    fitness_value, _, _, _, _ = calculate_population_fitness(
      key_for_fitness,  # This key needs to be managed carefully
      jnp.expand_dims(particle, axis=0),  # Make it a population of 1
      mpnn_model_instance,
      mpnn_model_is_active_static,
    )
    return fitness_value[0]  # TODO: make this better

  return population_fitness_fn, single_fitness_fn
