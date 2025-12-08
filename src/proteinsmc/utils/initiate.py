"""Utility functions to initialize sampler states."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp

from .translation import aa_to_nucleotide, nucleotide_to_aa

if TYPE_CHECKING:
  from jaxtyping import Int

  from proteinsmc.models.types import EvoSequence


def generate_template_population(
  initial_sequence: EvoSequence,
  population_size: Int | None,
  input_sequence_type: Literal["protein", "nucleotide"],
  output_sequence_type: Literal["protein", "nucleotide"] = "nucleotide",
) -> EvoSequence:
  """Create a JAX array representing a population from a single string sequence.

  This is a utility to easily create the `seed_sequences` needed for `SMCConfig`.

  Args:
      initial_sequence: The sequence as a string.
      population_size: The number of identical sequences in the population.
      input_sequence_type: The type of the initial sequence, either "protein" or "nucleotide".
      output_sequence_type: The type of the output sequences, either "protein" or "nucleotide".
          Defaults to "nucleotide".

  Returns:
      A JAX array of shape (population_size, sequence_length).

  """
  if population_size is None:
    population_size = jnp.array(1, dtype=jnp.int32)
  valid_types = ("protein", "nucleotide")
  if input_sequence_type not in valid_types:
    msg = "Invalid input_sequence_type"
    raise ValueError(msg)
  if output_sequence_type not in valid_types:
    msg = "Invalid output_sequence_type"
    raise ValueError(msg)

  if input_sequence_type == "protein":
    aa_seq = jnp.array(initial_sequence, dtype=jnp.int8)
    if aa_seq.ndim == 2 and aa_seq.shape[0] == 1:  # noqa: PLR2004
      aa_seq = jnp.squeeze(aa_seq, axis=0)
  elif input_sequence_type == "nucleotide":
    nuc_seq = jnp.array(initial_sequence, dtype=jnp.int8)
    if nuc_seq.ndim == 2 and nuc_seq.shape[0] == 1:  # noqa: PLR2004
      nuc_seq = jnp.squeeze(nuc_seq, axis=0)
    aa_seq, _ = nucleotide_to_aa(nuc_seq)

  if output_sequence_type == "protein":
    pop = jnp.tile(aa_seq, (population_size, 1))
  elif output_sequence_type == "nucleotide":
    nuc_seq, _ = aa_to_nucleotide(aa_seq)
    pop = jnp.tile(nuc_seq, (population_size, 1))

  return pop.astype(jnp.int8)
