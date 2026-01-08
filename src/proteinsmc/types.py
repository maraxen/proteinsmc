"""Type definitions for proteinsmc."""

from __future__ import annotations

from typing import Any, Literal, Protocol

import numpy as np
from jaxtyping import Array, Float, Int, PRNGKeyArray, UInt8

ArrayLike = Array | np.ndarray

# Scalar Types
Scalar = Int[ArrayLike, ""]
ScalarFloat = Float[ArrayLike, ""]

# Sequence Types
NucleotideSequence = Int[ArrayLike, "num_nucleotides"]
ProteinSequence = Int[ArrayLike, "num_residues"]
AminoAcidSequence = ProteinSequence  # alias
CodonSequence = Int[ArrayLike, "num_residues"]
OneHotAminoAcid = Float[ArrayLike, "num_residues 20"]

EvoSequence = Int[ArrayLike, "sequence_length"]
BatchEvoSequence = Int[ArrayLike, "batch_size sequence_length"]

SequenceType = Literal["nucleotide", "protein"]
UUIDArray = UInt8[ArrayLike, "36"]


# Protocols
class Vmapped(Protocol):
  """Protocol for vmapped functions."""

  def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    """Apply the vmapped function."""
    ...


VmappedFitnessFunc = Vmapped


class VmappedTranslation(Vmapped, Protocol):
  """Protocol for vmapped translation functions."""

  def __call__(self, sequences: BatchEvoSequence) -> BatchEvoSequence:
    """Translate a batch of sequences."""
    ...


# aliases
PRNGKey = PRNGKeyArray
