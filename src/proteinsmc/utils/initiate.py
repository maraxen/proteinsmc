"""Utility functions to generate a population of sequences from a single initial sequence."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp

from .constants import AA_CHAR_TO_INT_MAP, NUCLEOTIDES_INT_MAP
from .translation import reverse_translate, translate

if TYPE_CHECKING:
  from .types import PopulationSequences


def generate_template_population(
  initial_sequence: str,
  population_size: int,
  input_sequence_type: Literal["protein", "nucleotide"],
  output_sequence_type: Literal["protein", "nucleotide"] = "nucleotide",
) -> PopulationSequences:
  """Creates a JAX array representing a population from a single string sequence.

  This is a utility to easily create the `template_sequences` needed for `SMCConfig`.

  Args:
      initial_sequence: The sequence as a string.
      population_size: The number of identical sequences in the population.
      sequence_type: The type of sequence, 'protein' or 'nucleotide'.

  Returns:
      A JAX array of shape (population_size, sequence_length).

  """
  valid_types = ("protein", "nucleotide")
  if input_sequence_type not in valid_types:
    msg = "Invalid input_sequence_type"
    raise ValueError(msg)
  if output_sequence_type not in valid_types:
    msg = "Invalid output_sequence_type"
    raise ValueError(msg)

  if input_sequence_type == "protein":
    try:
      aa_seq = [AA_CHAR_TO_INT_MAP[res] for res in initial_sequence]
    except KeyError as e:
      msg = f"Invalid amino acid: {e.args[0]}"
      raise ValueError(msg) from e
    aa_seq = jnp.array(aa_seq, dtype=jnp.int8)
  elif input_sequence_type == "nucleotide":
    try:
      nuc_seq = [NUCLEOTIDES_INT_MAP[nuc] for nuc in initial_sequence]
    except KeyError as e:
      msg = f"Invalid nucleotide: {e.args[0]}"
      raise ValueError(msg) from e
    nuc_seq = jnp.array(nuc_seq, dtype=jnp.int8)
    aa_seq, _ = translate(nuc_seq)

  if output_sequence_type == "protein":
    pop = jnp.tile(aa_seq, (population_size, 1))
  elif output_sequence_type == "nucleotide":
    nuc_seq, _ = reverse_translate(aa_seq)
    pop = jnp.tile(nuc_seq, (population_size, 1))

  return pop.astype(jnp.int8)
