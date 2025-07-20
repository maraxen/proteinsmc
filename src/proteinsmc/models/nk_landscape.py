"""Data structures for NK landscape."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from flax.struct import PyTreeNode

if TYPE_CHECKING:
  from jaxtyping import Array, Float

  InteractionTable = Float[Array, "N K"]
  FitnessTable = Float[Array, "N q q ... q"]


@dataclass(frozen=True)
class NKLandscapeConfig:
  """Configuration for the NK landscape.

  Attributes:
      n: Number of sites (N).
      k: Number of neighbors (K).
      q: Number of states per site (q).

  """

  n: int = field(default=20)
  k: int = field(default=2)
  q: int = field(default=2)


class NKLandscape(PyTreeNode):
  """Stores the complete NK landscape: interactions and fitness tables.

  Attributes:
      interactions: (N, K) array of neighbor indices, padded with -1.
      fitness_tables: (N, q, q, ..., q) array of fitness contributions for each site and state.

  """

  interactions: InteractionTable
  fitness_tables: FitnessTable
