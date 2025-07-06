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

    # Determine all possible neighbors for site 'i' (all sites except 'i' itself)
    # The size argument ensures a static shape for the output of setdiff1d
    possible_neighbors_for_site_i = jnp.setdiff1d(
      sites, jnp.array([i]), assume_unique=True, size=N - 1 if N > 1 else 0
    )
    # Get the actual number of possible neighbors for site 'i'
    num_actually_possible_for_site_i = possible_neighbors_for_site_i.shape[0]

    # Determine the number of neighbors to select for site 'i'.
    # This is the minimum of K (max desired neighbors) and the actual         # number possible.
    num_to_select_for_site_i = jnp.minimum(K, num_actually_possible_for_site_i)

    # Shuffle the list of actual possible neighbors to ensure random selection
    permuted_actual_possible_neighbors = jax_random.permutation(
      key_site_i, possible_neighbors_for_site_i
    )

    # Now, construct the interaction row for site 'i'. This row will         # have a fixed size K.
    # It will contain the first 'num_to_select_for_site_i' elements
    # from 'permuted_actual_possible_neighbors', and the rest will be -1 (padding).

    # Create a new row for site 'i', initialized with the padding value -1
    new_interaction_row_for_site_i = jnp.full(K, -1, dtype=jnp.int32)

    # Define the body function for an inner loop that populates the K         # slots in the new_interaction_row_for_site_i
    def assign_kth_neighbor_in_row(k_idx, current_row_values_carry):
      value_to_assign = lax.cond(
        k_idx < num_to_select_for_site_i,
        lambda p_neighbors: p_neighbors[k_idx],  # Get the actual
        lambda p_neighbors: -1,  # Otherwise, assign padding value
        permuted_actual_possible_neighbors,  # Pass the permuted neighbors array to the lambda
      )
      return current_row_values_carry.at[k_idx].set(value_to_assign)

    # Populate the new_interaction_row_for_site_i by looping K times
    populated_row = lax.fori_loop(0, K, assign_kth_neighbor_in_row, new_interaction_row_for_site_i)

    # Update the i-th row of the main interactions_carry table with the populated_row
    return interactions_carry.at[i].set(populated_row)

  # Loop over all N sites to generate their interactions
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
  # Create a unique key for this specific interaction
  # Fold in site_idx, its state, and neighbor states
  # For JAX, inputs to hash/fold_in should be JAX types, preferably integers or arrays.
  # Ensure neighbor_states has a fixed shape, (K,)

  interaction_key_base = jax_random.fold_in(landscape_master_key, site_idx)
  interaction_key_base = jax_random.fold_in(interaction_key_base, site_state)

  # Iteratively fold in neighbor states
  # Using a loop for folding, though for small K direct folding is also possible
  def fold_neighbor_states(idx, current_key):
    return jax_random.fold_in(current_key, neighbor_states[idx])

  # We need fixed iterations for lax.fori_loop for K neighbors
  # neighbor_states is expected to be of length K
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

    # Gather neighbor states, handling padding in interactions_map
    # If neighbor_indices_for_site[j] is -1 (padding), use a default state (e.g., 0)
    # This default state should ideally not affect the contribution in a meaningful way
    # or be filtered out if the contribution function can handle variable K.
    # For simplicity, we'll pass them, and get_nk_site_contribution_jax can be designed
    # to ignore contributions from padded neighbors if needed, or use their state (e.g. 0).

    # A simple way to handle -1 indices for neighbors is to map them to a valid index
    # (e.g. the site itself or index 0) and hope the contribution function is robust
    # or simply pass them and ensure the contribution function uses them in hashing.
    # For robustness, let's assume the contribution function is designed for fixed K input.
    # If an interaction is -1, it means no *actual* Kth neighbor.
    # A common approach is to ensure neighbor_states passed to contribution has fixed size K.

    # Initialize neighbor_states array, e.g. with zeros or a special padding value
    actual_neighbor_states = jnp.zeros(K, dtype=jnp.int32)

    def get_neighbor_state(k_idx, states_arr):
      neighbor_idx = neighbor_indices_for_site[k_idx]
      # If neighbor_idx is valid (>=0), get its state from config. Otherwise, keep padding (e.g. 0).
      return jax.lax.cond(
        neighbor_idx >= 0,
        lambda cfg, n_idx: cfg[n_idx],
        lambda cfg, n_idx: 0,  # Default state for padded/non-existent neighbor
        config,
        neighbor_idx,
      )

    # Populate actual_neighbor_states, assuming K is fixed for the loop
    # This loop will run K times.
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

  return total_fitness / N  # Average fitness


# Vmapped version for populationes of particles
@partial(jit, static_argnames=("N", "K", "q"))
def calculate_nk_fitness_population_jax(
  configs_population, interactions_map, landscape_master_key, N, K, q
):
  return vmap(calculate_nk_fitness_single_jax, in_axes=(0, None, None, None, None, None))(
    configs_population, interactions_map, landscape_master_key, N, K, q
  )
