"""JAX-based utilities for generating and evaluating NK fitness landscapes.

This module provides functions to generate NK interaction maps, compute site contributions,
and calculate fitness for single configurations or populations using JAX for efficiency.

Refactored to optimize calculate_nk_fitness_single and add
generate_generalized_nk_interactions.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from jax import random as jax_random
from jax import vmap

from proteinsmc.models.nk_landscape import NKLandscape

if TYPE_CHECKING:
  from jaxtyping import PRNGKeyArray

from jaxtyping import Array, Float, Int

InteractionTable = Int[Array, "N K"]
FitnessTable = Float[Array, "N q q ... q"]
NKInput = Int[Array, "N"]
NKPopulation = Int[Array, "P N"]


@dataclass
class InteractionCarry:
  """Stores the interaction carry for JAX loops.

  This is used to carry the state of the interactions array during the loop.

  Attributes:
      interactions: (N, K) array of neighbor indices, padded with -1.

  """

  interactions: InteractionTable


if TYPE_CHECKING:

  def generate_nk_interactions(
    key: PRNGKeyArray,
    n: int,
    k: int,
  ) -> InteractionTable:
    """Generate the interaction map for an NK model."""
    ...
else:

  @partial(jax.jit, static_argnames=("n", "k"))
  def generate_nk_interactions(
    key: PRNGKeyArray,
    n: int,
    k: int,
  ) -> InteractionTable:
    """Generate the interaction map for an NK model in JAX (Optimized).

    Args:
        key: JAX PRNG key.
        n: Number of sites (N).
        k: Number of neighbors (K).

    Returns:
        An (N, K) array of neighbor indices.

    """
    if k == 0:
      return jnp.full((n, k), -1, dtype=jnp.int32)

    sites = jnp.arange(n)
    possible_neighbors = jnp.array([jnp.roll(sites, -i - 1)[:-1] for i in range(n)])

    def _select_k_neighbors(key_site: PRNGKeyArray, site_neighbors: Int) -> Int:
      """Select k random neighbors for a single site."""
      return jax.random.choice(key_site, site_neighbors, shape=(k,), replace=False)

    keys = jax.random.split(key, n)
    return vmap(_select_k_neighbors)(keys, possible_neighbors)


@partial(jax.jit, static_argnames=("n", "max_k", "p_connect"))
def generate_generalized_nk_interactions(
  key: PRNGKeyArray,
  n: int,
  max_k: int,
  p_connect: float,
) -> InteractionTable:
  """Generate a generalized NK interaction map (Optimized).

  Args:
      key: JAX PRNG key.
      n: Number of sites (N).
      max_k: The maximum possible number of external neighbors.
      p_connect: The probability of connection for each potential neighbor.

  Returns:
      An (N, max_k) array of neighbor indices, padded with -1.

  """
  if max_k == 0:
    return jnp.full((n, max_k), -1, dtype=jnp.int32)

  sites = jnp.arange(n)
  possible_neighbors = jnp.array([jnp.roll(sites, -i - 1)[:-1] for i in range(n)])

  def _select_for_site(key_site: PRNGKeyArray, neighbors_for_site: Int) -> Int:
    """Select up to max_k neighbors for one site based on p_connect."""
    key_connect, key_shuffle = jax.random.split(key_site)

    # Select neighbors
    is_selected = jax.random.bernoulli(key_connect, p_connect, shape=(n - 1,))
    selected_neighbors = jnp.array(jnp.where(is_selected, neighbors_for_site, -1))

    # Shuffle selected neighbors
    shuffled_selected = jax.random.permutation(key_shuffle, selected_neighbors)

    # Move -1s to the end
    sorted_neighbors = jnp.roll(shuffled_selected, jnp.sum(shuffled_selected == -1))

    # Pad to max_k
    padded_neighbors = jnp.pad(
      sorted_neighbors, (0, max_k - (n - 1)), "constant", constant_values=-1
    )

    return padded_neighbors[:max_k]

  keys = jax.random.split(key, n)
  return vmap(_select_for_site)(keys, possible_neighbors)


def generate_nk_model(
  key: PRNGKeyArray,
  n: int,
  k: int,
  q: int,
) -> NKLandscape:
  """Generate a full NK (Potts) model with pre-computed fitness tables.

  Args:
      key: JAX PRNG key.
      n: Number of sites (N).
      k: Number of neighbors (K).
      q: Number of states per site (q).

  Returns:
      An NKLandscape object containing:
          - interactions: (N, K) array of neighbor indices, padded with -1.
          - fitness_tables: (N, q, q, ..., q) array of fitness contributions
            for each site and state, where the shape is (N, q, ..., q)
            with K+1 dimensions of size q.

  """
  key_interactions, key_fitness = jax_random.split(key)

  interactions = generate_nk_interactions(key_interactions, n, k)

  fitness_table_shape = (n, *([q] * (k + 1)))
  fitness_tables = jax_random.uniform(key_fitness, shape=fitness_table_shape)

  return NKLandscape(interactions=interactions, fitness_tables=fitness_tables)


if TYPE_CHECKING:

  def calculate_nk_fitness_single(
    single_sequence: NKInput,
    landscape: NKLandscape,
    n: int,
    k: int,
  ) -> Float:
    """Calculate fitness for one configuration."""
    ...
else:

  @partial(jax.jit, static_argnames=("n", "k"))
  def calculate_nk_fitness_single(
    single_sequence: NKInput,
    landscape: NKLandscape,
    n: int,
    k: int,
  ) -> Float:
    """Calculate fitness for one configuration (Refined).

    Args:
        single_sequence: A single sequence of shape (N,).
        landscape: NKLandscape containing interactions and fitness tables.
        n: Number of sites (N).
        k: Number of neighbors (K).

    Returns:
        Mean fitness contribution across all sites.

    """
    site_indices = jnp.arange(n)
    focal_states = single_sequence
    neighbor_site_indices = landscape.interactions
    neighbor_states = single_sequence[neighbor_site_indices]
    lookup_indices = jnp.stack(
      [site_indices, focal_states, *[neighbor_states[:, i] for i in range(k)]]
    )
    all_contributions = landscape.fitness_tables[tuple(lookup_indices)]
    return jnp.mean(all_contributions)


if TYPE_CHECKING:

  def calculate_nk_fitness_population(
    population: NKPopulation,
    landscape: NKLandscape,
    n: int,
    k: int,
  ) -> Float:
    """Calculate fitness for a population via vmap."""
    ...
else:

  @partial(jax.jit, static_argnames=("n", "k"))
  def calculate_nk_fitness_population(
    population: NKPopulation,
    landscape: NKLandscape,
    n: int,
    k: int,
  ) -> Float:
    """Calculate fitness for a population via vmap."""
    return vmap(calculate_nk_fitness_single, in_axes=(0, None, None, None))(
      population,
      landscape,
      n,
      k,
    )
