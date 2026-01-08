"""Padding and masking utilities for static-shape JAX compilation in ProteinSMC.

Provides fixed sizes for population and sequence length to avoid XLA recompilation
during SMC sampling when population sizes change.

Constants:
    - MAX_POPULATION: 512 particles
    - MAX_SEQ_LEN: 1200 residues
    - SEQ_BUCKETS: (200, 400, 800, 1200)

All populations are padded to fixed sizes, with valid_particle_mask used to
exclude invalid particles from computations.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

# Fixed size constants for SMC sampling
MAX_POPULATION: int = 512
MAX_SEQ_LEN: int = 1200

# Sequence length buckets
SEQ_BUCKETS: tuple[int, ...] = (200, 400, 800, 1200)


def get_seq_bucket(n_residues: int) -> int:
  """Find the smallest bucket that can fit the sequence length.

  Args:
      n_residues: Actual number of residues.

  Returns:
      The bucket size to use.

  Raises:
      ValueError: If length exceeds all buckets.
  """
  for bucket in SEQ_BUCKETS:
    if n_residues <= bucket:
      return bucket
  raise ValueError(f"Sequence length {n_residues} exceeds all buckets {SEQ_BUCKETS}")


def pad_population(
  sequences: Array,
  target_population: int = MAX_POPULATION,
) -> Array:
  """Pad a population of sequences to the target size.

  Args:
      sequences: Population array of shape (n_particles, seq_len).
      target_population: Target padded population size.

  Returns:
      Padded population of shape (target_population, seq_len).
  """
  real_pop = sequences.shape[0]
  n_pad = target_population - real_pop
  if n_pad > 0:
    return jnp.pad(sequences, ((0, n_pad), (0, 0)), constant_values=0)
  return sequences[:target_population]


def pad_sequence(sequence: Array, target_len: int) -> Array:
  """Pad a sequence to the target length.

  Args:
      sequence: Sequence array of shape (seq_len,) or (batch, seq_len).
      target_len: Target padded length.

  Returns:
      Padded sequence.
  """
  if sequence.ndim == 1:
    n_pad = target_len - sequence.shape[0]
    return jnp.pad(sequence, (0, n_pad), constant_values=0)
  # Batch case: (batch, seq_len)
  n_pad = target_len - sequence.shape[1]
  return jnp.pad(sequence, ((0, 0), (0, n_pad)), constant_values=0)


def create_particle_mask(
  real_population: int, max_population: int = MAX_POPULATION
) -> Bool[Array, " max_population"]:
  """Create a mask for valid particles in a population.

  Args:
      real_population: Actual number of valid particles.
      max_population: Maximum padded population size.

  Returns:
      Boolean mask of shape (max_population,), True for valid particles.
  """
  return jnp.arange(max_population) < real_population


def create_sequence_mask(real_len: int, padded_len: int) -> Bool[Array, " padded_len"]:
  """Create a mask for valid sequence positions.

  Args:
      real_len: Actual sequence length.
      padded_len: Padded sequence length.

  Returns:
      Boolean mask of shape (padded_len,), True for valid positions.
  """
  return jnp.arange(padded_len) < real_len


def masked_mean(values: Array, mask: Array) -> Float[Array, ""]:
  """Compute mean over valid (masked) elements.

  Args:
      values: Array of values.
      mask: Boolean mask (True for valid elements).

  Returns:
      Mean over valid elements.
  """
  return jnp.sum(values * mask) / jnp.sum(mask)


def masked_sum(values: Array, mask: Array) -> Float[Array, ""]:
  """Compute sum over valid (masked) elements.

  Args:
      values: Array of values.
      mask: Boolean mask (True for valid elements).

  Returns:
      Sum over valid elements.
  """
  return jnp.sum(values * mask)
