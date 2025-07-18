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

from proteinsmc.models.nk_landscape import NKLandscape

if TYPE_CHECKING:
    from jaxtyping import PRNGKeyArray

from jaxtyping import Array, Float, Int

InteractionTable = Float[Array, "N K"]
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
) -> InteractionTable:
    """Generate the interaction map for an NK model in JAX.

    Args:
        key: JAX PRNG key.
        n: Number of sites (N).
        k: Number of neighbors (K).

    Returns:
        An (N, K) array of neighbor indices, padded with -1 if K > N-1.
        Each row contains K unique indices of neighbors for each site, or -1 for padding.

    """
    if k == 0:
        return jnp.full((n, k), -1, dtype=jnp.int32)

    sites = jnp.arange(n)

    def _get_neighbors(i: Int) -> Int:
        """Get k neighbors for a single site, excluding itself."""
        doubled_sites = jnp.concatenate([sites, sites])
        return lax.dynamic_slice(doubled_sites, [i + 1], [n - 1])

    possible_neighbors = vmap(_get_neighbors)(sites)

    def _select_for_site(key_site: PRNGKeyArray, neighbors_for_site: Int) -> Int:
        """Select k neighbors for a single site."""
        num_potential = n - 1
        shuffled = jax.random.permutation(key_site, neighbors_for_site)
        padding = jnp.full(max(0, k - num_potential), -1, dtype=jnp.int32)
        padded_shuffled = jnp.concatenate([shuffled, padding])
        return padded_shuffled[:k]

    keys = jax_random.split(key, n)
    return vmap(_select_for_site)(keys, possible_neighbors)


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
def calculate_nk_fitness_single(
    single_sequence: NKInput,
    landscape: NKLandscape,
    n: int,
    k: int,
) -> Float:
    """Calculate fitness for one configuration using pre-computed tables."""
    if k == 0:
        indices = (jnp.arange(n), single_sequence)
        return jnp.mean(landscape.fitness_tables[indices])

    def get_contribution_for_site(i: Int) -> Float:
        """Calculate the fitness contribution for a single site i."""
        focal_site_state = single_sequence[i]
        neighbor_indices = landscape.interactions[i]

        neighbor_states = jnp.where(
            neighbor_indices != -1,
            single_sequence[neighbor_indices],
            0,
        )

        lookup_indices = (i, focal_site_state, *[neighbor_states[j] for j in range(k)])
        return landscape.fitness_tables[lookup_indices]

    all_contributions = vmap(get_contribution_for_site)(jnp.arange(n))

    return jnp.mean(all_contributions)


@partial(jit, static_argnames=("n", "k"))
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
