"""JAX-based utilities for generating and evaluating NK fitness landscapes.

This module provides functions to generate NK interaction maps, compute site contributions,
and calculate fitness for single configurations or populations using JAX for efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import jit, lax, vmap
from jax import random as jax_random

if TYPE_CHECKING:
  from jaxtyping import PRNGKeyArray

from jaxtyping import Array, Float, Int

InteractionTable = Float[Array, "N K"]
FitnessTable = Float[Array, "N q q ... q"]
NKInput = Int[Array, "N"]
NKPopulation = Int[Array, "P N"]


@dataclass(frozen=True)
class NKLandscape:
  """Stores the complete NK landscape: interactions and fitness tables.

  Attributes:
    interactions: (N, K) array of neighbor indices, padded with -1.
    fitness_tables: (N, q, q, ..., q) array of fitness contributions for each site and state.

  """

  interactions: InteractionTable
  fitness_tables: FitnessTable

  def tree_flatten(self) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility."""
    children = (self.interactions, self.fitness_tables)
    return children, {}

  @classmethod
  def tree_unflatten(cls, _aux_data: dict, children: tuple) -> NKLandscape:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(interactions=children[0], fitness_tables=children[1])


@dataclass
class InteractionCarry:
  """Stores the interaction carry for JAX loops.

  This is used to carry the state of the interactions array during the loop.

  Attributes:
    interactions: (N, K) array of neighbor indices, padded with -1.

  """

  interactions: InteractionTable

  def tree_flatten(self) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility."""
    return (self.interactions,), {}

  @classmethod
  def tree_unflatten(cls, _aux_data: dict, children: tuple) -> InteractionCarry:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(interactions=children[0])


@partial(jit, static_argnames=("n", "k"))
def generate_nk_interactions(
  key: PRNGKeyArray,
  n: int,
  k: int,
) -> jax.Array:
  """Generate the interaction map for an NK model in JAX.

  This function determines which K sites influence each of the N sites.

  Args:
    key: JAX PRNG key.
    n: Number of sites (N).
    k: Number of neighbors (K).

  Returns:
    An (N, K) array of neighbor indices. If K=0, returns an empty array of
    the correct shape. If a site has fewer than K possible neighbors (i.e., N-1 < K),
    the row is padded with -1.

  """
  if k == 0:
    return jnp.full((n, k), -1, dtype=jnp.int32)

  interactions = jnp.full((n, k), -1, dtype=jnp.int32)
  sites = jnp.arange(n)

  def body_fun_n(
    i: int,
    interactions_carry: InteractionCarry,
  ) -> InteractionCarry:
    """Populate the interaction row for a single site i.

    Args:
      i: Index of the site (0 to N-1).
      interactions_carry: Current state of the interactions array.

    Returns:
      Updated interactions array with the row for site i filled in.

    """
    key_site_i = jax_random.fold_in(key, i)

    possible_neighbors = jnp.setdiff1d(
      sites,
      jnp.array([i]),
      assume_unique=True,
      size=n - 1,
      fill_value=-1,
    )
    possible_neighbors = possible_neighbors[possible_neighbors != -1]
    num_possible = possible_neighbors.shape[0]

    num_to_select = jnp.minimum(k, num_possible)

    selected_neighbors = jax_random.choice(
      key_site_i,
      possible_neighbors,
      shape=(int(num_to_select),),
      replace=False,
    )

    padded_neighbors = jnp.full((k,), -1, dtype=jnp.int32)
    padded_neighbors = padded_neighbors.at[:num_to_select].set(selected_neighbors)
    interactions_carry.interactions.at[i].set(padded_neighbors)

    return interactions_carry

  return lax.fori_loop(0, n, body_fun_n, interactions)


@partial(jit, static_argnames=("n", "k", "q"))
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


@partial(jit, static_argnames=("n", "k"))
def calculate_nk_fitness_single_jax(
  single_sequence: NKInput,
  landscape: NKLandscape,
  n: int,
  k: int,
) -> Float:
  """Calculate fitness for one configuration using pre-computed tables."""

  def site_contribution_loop_body(i: int, _: jax.Array) -> tuple[int, Float]:
    """Calculate contribution of site i to the total fitness."""
    focal_site_state = single_sequence[i]
    neighbor_indices = landscape.interactions[i]
    neighbor_states = single_sequence[neighbor_indices]

    lookup_indices = (i, focal_site_state, *[neighbor_states[j] for j in range(k)])
    return 0, landscape.fitness_tables[lookup_indices]

  _, site_contributions = lax.scan(site_contribution_loop_body, 0, jnp.arange(n))
  return jnp.mean(site_contributions)


@partial(jit, static_argnames=("n", "k"))
def calculate_nk_fitness_population_jax(
  population: NKPopulation,
  landscape: NKLandscape,
  n: int,
  k: int,
) -> Float:
  """Calculate fitness for a population via vmap."""
  return vmap(calculate_nk_fitness_single_jax, in_axes=(0, None, None, None))(
    population,
    landscape,
    n,
    k,
  )
