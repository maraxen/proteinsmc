"""NK landscape scoring factory for integration with the fitness registry.

Provides a `make_nk_score` factory used by `get_fitness_function` to create
per-sequence scoring functions that evaluate NK landscapes.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from proteinsmc.utils.nk_landscape import NKLandscape, calculate_nk_fitness_single

if TYPE_CHECKING:
  from collections.abc import Callable

  from jaxtyping import Array, PRNGKeyArray, PyTree


def make_nk_score(landscape: NKLandscape, n: int, k: int) -> Callable:
  """Return a fitness function for an NK landscape.

  Accepts keyword args produced by the `FitnessFunction.kwargs` in the
  project. For backward compatibility this function will also accept keys
  named 'N' and 'K'.
  """
  fitness_fn = partial(calculate_nk_fitness_single, landscape=landscape, n=n, k=k)

  def nk_score(
    sequence: Array,
    _key: PRNGKeyArray | None = None,
    _context: PyTree | Array | None = None,
  ) -> Array:
    """Calculate fitness for a single sequence using the NK landscape."""
    return fitness_fn(sequence)

  return nk_score
