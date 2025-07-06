from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax, vmap
from jax import random as jax_random


@partial(jit, static_argnames=("N", "K", "q"))
def generate_nk_interactions_jax(key, N, K, q):
  """
  Generates the interaction map for an NK model in JAX.
  Corrected to handle dynamic slicing issues for JIT.
  Returns an (N, K) array of neighbor indices, padded with -1.
  """
  if K == 0:
    return jnp.full((N, K), -1, dtype=jnp.int32)

  interactions = jnp.full((N, K), -1, dtype=jnp.int32)
  sites = jnp.arange(N)

  def body_fun_n(i, interactions_carry):
    key_site_i = jax_random.fold_in(key, i)

    possible_neighbors_for_site_i = jnp.setdiff1d(
      sites, jnp.array([i]), assume_unique=True, size=N - 1 if N > 1 else 0
    )
    num_actually_possible_for_site_i = possible_neighbors_for_site_i.shape[0]

    num_to_select_for_site_i = jnp.minimum(K, num_actually_possible_for_site_i)

    permuted_actual_possible_neighbors = jax_random.permutation(
      key_site_i, possible_neighbors_for_site_i
    )

    new_interaction_row_for_site_i = jnp.full(K, -1, dtype=jnp.int32)

    def assign_kth_neighbor_in_row(k_idx, current_row_values_carry):
      value_to_assign = lax.cond(
        k_idx < num_to_select_for_site_i,
        lambda p_neighbors: p_neighbors[k_idx],
        lambda p_neighbors: -1,
        permuted_actual_possible_neighbors,
      )
      return current_row_values_carry.at[k_idx].set(value_to_assign)

    populated_row = lax.fori_loop(0, K, assign_kth_neighbor_in_row, new_interaction_row_for_site_i)

    return interactions_carry.at[i].set(populated_row)

  final_interactions = lax.fori_loop(0, N, body_fun_n, interactions)
  return final_interactions


@partial(jit, static_argnames=("K_plus_1",))
def get_nk_site_contribution_jax(
  site_idx, site_state, neighbor_states, landscape_master_key, K_plus_1
):
  """
  Generates a deterministic, pseudo-random contribution for a given site
  and its state configuration using JAX PRNG.
  neighbor_states should be padded with a common value (e.g., 0 or -1) if actual neighbors < K.
  Ensure this padding is consistent.
  """

  interaction_key_base = jax_random.fold_in(landscape_master_key, site_idx)
  interaction_key_base = jax_random.fold_in(interaction_key_base, site_state)

  def fold_neighbor_states(idx, current_key):
    return jax_random.fold_in(current_key, neighbor_states[idx])

  final_interaction_key = jax.lax.fori_loop(
    0, neighbor_states.shape[0], fold_neighbor_states, interaction_key_base
  )

  return jax_random.uniform(final_interaction_key, shape=(), minval=0.0, maxval=1.0)


@partial(jit, static_argnames=("N", "K", "q"))
def calculate_nk_fitness_single_jax(config, interactions_map, landscape_master_key, N, K, q):
  """
  Calculates the average fitness of a given configuration on the fly using JAX.
  config: (N,) array of site states (0 to q-1)
  interactions_map: (N, K) array of neighbor indices (padded with -1)
  """
  if N == 0:
    return 0.0

  total_fitness = 0.0

  def body_fun_site(i, current_total_fitness_carry):
    site_state = config[i]
    neighbor_indices_for_site = interactions_map[i, :]  # Shape (K,)

    actual_neighbor_states = jnp.zeros(K, dtype=jnp.int32)

    def get_neighbor_state(k_idx, states_arr):
      neighbor_idx = neighbor_indices_for_site[k_idx]
      return jax.lax.cond(
        neighbor_idx >= 0,
        lambda cfg, n_idx: cfg[n_idx],
        lambda cfg, n_idx: 0,
        config,
        neighbor_idx,
      )

    actual_neighbor_states = jax.lax.fori_loop(
      0,
      K,
      lambda k_idx, states_arr_carry: states_arr_carry.at[k_idx].set(
        get_neighbor_state(k_idx, states_arr_carry)
      ),
      actual_neighbor_states,
    )

    site_contribution = get_nk_site_contribution_jax(
      i, site_state, actual_neighbor_states, landscape_master_key, K + 1
    )
    return current_total_fitness_carry + site_contribution

  total_fitness = jax.lax.fori_loop(0, N, body_fun_site, total_fitness)

  return total_fitness / N


@partial(jit, static_argnames=("N", "K", "q"))
def calculate_nk_fitness_population_jax(
  configs_population, interactions_map, landscape_master_key, N, K, q
):
  return vmap(calculate_nk_fitness_single_jax, in_axes=(0, None, None, None, None, None))(
    configs_population, interactions_map, landscape_master_key, N, K, q
  )
