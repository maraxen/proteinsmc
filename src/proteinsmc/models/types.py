"""Core types used throughout the proteinsmc library.

It uses jaxtyping for type annotations of JAX arrays, which provides
shape and dtype information in the type hints. This is crucial for
maintaining clarity and correctness in a JAX-based codebase.
"""

from __future__ import annotations

import enum
from typing import Protocol, TypeVar

import jax.numpy as jnp
from colabdesign.mpnn import mk_mpnn_model
from jaxtyping import Array, Int

NucleotideSequence = Int[Array, "sequence_length 4"]
"""A sequence of nucleotides represented as integers.
The integers correspond to the four nucleotides (A, T, C, G).
"""
ProteinSequence = Int[Array, "sequence_length alphabet_size"]
"""A sequence of amino acids represented as integers.
The integers correspond to the amino acids in the protein alphabet.
"""
EvoSequence = Int[NucleotideSequence | ProteinSequence, "sequence_length alphabet_size"]
BatchEvoSequence = Int[Array, "batch_size sequence_length alphabet_size"]
MPNNModel = mk_mpnn_model


class SequenceType(enum.Enum):
  """Enumeration for sequence types.

  This is used to specify the type of sequence being processed.
  """

  NUCLEOTIDE = "nucleotide"
  PROTEIN = "protein"


class Vmapped(Protocol):
  """A protocol for functions that can be vmapped.

  This is used to type hint functions that are expected to be vectorized.
  """

  def __call__(self, *args, **kwargs) -> ...: ...  # noqa: D102, ANN002, ANN003


T = TypeVar("T")
VmappedFitnessFunc = Vmapped


class VmappedTranslation(Vmapped, Protocol):
  """A protocol for functions that can be vmapped.

  This is used to type hint functions that are expected to be vectorized.
  """

  def __call__(self, sequences: BatchEvoSequence) -> jnp.ndarray: ...  # noqa: D102
