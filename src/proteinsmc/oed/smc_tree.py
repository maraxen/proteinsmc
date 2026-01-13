"""SMC-based tree evolution logic for calculating evolutionary trajectories."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, PRNGKeyArray, PyTree
from trex import nk_model
from trex.types import Adjacency
from trex.utils.memory import safe_map
from trex.utils.types import EvoSequence

from proteinsmc.utils import mutation
from proteinsmc.utils.jax_utils import chunked_map


def evolve_path_smc(
  _key: PRNGKeyArray,
  initial_population: Int[Array, "pop_size seq_len"],
  path_indices: Int[Array, total_steps],
  parent_key_array: PRNGKeyArray,
  landscape: PyTree,
  mutation_rate: float,
  n_states: int,
  selection_intensity: float = 1.0,
  inference_batch_size: int = 64,
) -> tuple[Int[Array, "total_steps pop_size seq_len"], Int[Array, "total_steps pop_size"]]:
  """Evolve a population along a specified path of edges using SMC.

  Crucially, this function ensures that shared branches in a tree use identical
  RNG keys (derived from `master_key_array` and `path_indices`), enabling
  redundant parallelism where shared history is exactly reproduced.

  Args:
      _key: Initial PRNG key (unused for evolution steps to ensure
            path-determinism, but kept for API consistency).
      initial_population: Starting population of sequences.
      path_indices: Array of edge indices defining the path to traverse.
                    At each step `t`, `path_indices[t]` points to the
                    specific edge in `master_key_array` to use.
      parent_key_array: Array of pre-generated keys for all edges in the
                        tree structure. Shape (num_edges, 2).
      landscape: NK landscape PyTree for fitness evaluation.
      mutation_rate: Probability of mutation per site.
      n_states: Number of possible states for each site (e.g., 4 for DNA).
      selection_intensity: Exponent for fitness weighting (default 1.0).
      inference_batch_size: Batch size for fitness calculation.

  Returns:
      A tuple containing:
        - populations: History of resampled populations (total_steps, pop_size, seq_len).
        - ancestry: Resampling indices for each step (total_steps, pop_size).

  """
  # Ensure population is int8 to match mutation output
  population_carry = initial_population.astype(jnp.int8)

  total_steps = path_indices.shape[0]
  time_steps = jnp.arange(total_steps)

  # Helper for vectorized fitness
  def get_fitness_wrapper(seq: Array, landscape: PyTree) -> Array:
    return nk_model.get_fitness(seq, landscape)

  def scan_body(
    population: Int[Array, "pop_size seq_len"], inputs: tuple[int, int]
  ) -> tuple[Int[Array, "pop_size seq_len"], None]:
    edge_index, t = inputs
    edge_key_base = parent_key_array[edge_index]

    # Combine edge_key with step t for uniqueness across time on same edge
    # We use fold_in to mix the step into the key.
    step_key = jax.random.fold_in(edge_key_base, t)

    # Split for Mutation and Resampling
    key_mut, key_resample = jax.random.split(step_key)

    # 1. Mutation
    mutated_population = mutation.mutate(key_mut, population, mutation_rate, n_states)

    # 2. Selection / Resampling
    # Calculate fitness using chunked_map for memory efficiency
    fitness_scores = chunked_map(
      get_fitness_wrapper,
      mutated_population,
      batch_size=inference_batch_size,
      static_args={"landscape": landscape},
    )

    # Use fitness directly as weights
    weights = fitness_scores**selection_intensity
    weights = jnp.maximum(weights, 1e-10)

    # Normalize
    weights = weights / jnp.sum(weights)

    # Resample
    # Multinomial resampling
    pop_size = population.shape[0]
    chosen_indices = jax.random.choice(
      key_resample, pop_size, shape=(pop_size,), p=weights, replace=True
    )
    resampled_population = mutated_population[chosen_indices]

    return resampled_population, (resampled_population, chosen_indices)

  # Run scan
  _, (history, ancestry) = jax.lax.scan(scan_body, population_carry, (path_indices, time_steps))

  return history, ancestry


def _run_smc_path_with_traceback(
  path: Int[Array, " T"],
  initial_pop: Int[Array, "pop_size seq_len"],
  node_keys: PRNGKeyArray,
  landscape: PyTree,
  mutation_rate: float,
  n_states: int,
  selection_intensity: float,
  inference_batch_size: int,
  key: PRNGKeyArray,
) -> Int[Array, "T seq_len"]:
  """Evolve a single path and trace back the best lineage."""
  # The padding -1 should index into node_keys[-1] which is fine if we add a dummy key.
  safe_node_keys = jnp.concatenate([node_keys, node_keys[0:1]], axis=0)  # Add dummy at -1
  history, ancestry = evolve_path_smc(
    _key=key,  # Not used for evolution steps
    initial_population=initial_pop,
    path_indices=path,
    parent_key_array=safe_node_keys,
    landscape=landscape,
    mutation_rate=mutation_rate,
    n_states=n_states,
    selection_intensity=selection_intensity,
    inference_batch_size=inference_batch_size,
  )

  # Traceback logic to extract a single representative lineage
  total_steps = history.shape[0]

  # Calculate fitness of final population at the leaf
  final_pop = history[-1]
  final_fitness = chunked_map(
    lambda seq, ls: nk_model.get_fitness(seq, ls),
    final_pop,
    batch_size=inference_batch_size,
    static_args={"ls": landscape},
  )
  winner_idx = jnp.argmax(final_fitness)

  # Scan backwards to reconstruct the lineage indices
  def traceback_step(curr_idx: int, t: int) -> tuple[int, int]:
    # ancestry[t, curr_idx] is the parent index in history[t-1]
    parent_idx = ancestry[t, curr_idx]
    return parent_idx, curr_idx

  # We want indices for steps [total_steps-1, ..., 0]
  # We use a scan on reversed time
  _, lineage_indices = jax.lax.scan(traceback_step, winner_idx, jnp.arange(total_steps - 1, -1, -1))
  # lineage_indices is now [winner_idx, p_final-1, p_final-2, ..., p_1]
  # Reverse it back
  lineage_indices = jnp.flip(lineage_indices)

  # Extract sequences using reconstructed indices
  def extract_seq(t: int) -> Int[Array, " seq_len"]:
    return history[t, lineage_indices[t]]

  return jax.vmap(extract_seq)(jnp.arange(total_steps))


def generate_tree_data_smc(
  landscape: PyTree,
  adjacency: Adjacency,
  root_sequence: EvoSequence,
  mutation_rate: float,
  key: PRNGKeyArray,
  pop_size: int = 100,
  branch_length: int = 1,
  n_states: int = 20,
  selection_intensity: float = 1.0,
  inference_batch_size: int = 64,
) -> nk_model.PhylogeneticTree:
  """Generate tree data using SMC with redundant parallelism.

  Args:
      landscape: NK landscape PyTree.
      adjacency: Adjacency matrix (N, N).
      root_sequence: Starting sequence.
      mutation_rate: Mutation probability per site.
      key: RNG key.
      pop_size: Number of particles in SMC.
      branch_length: Number of steps per edge.
      n_states: Alphabet size.
      selection_intensity: Exponent for fitness weighting.
      inference_batch_size: Batch size for fitness calculation.

  Returns:
      PhylogeneticTree containing sequences for all nodes.

  """
  n_nodes = adjacency.shape[0]
  seq_len = root_sequence.shape[0]

  # 1. Topology Analysis
  # BFS to find parent indices and sorted nodes (from trex.nk_model)
  parent_indices = jnp.argmax(adjacency, axis=1)
  root_node = jnp.where(parent_indices == jnp.arange(n_nodes), size=1)[0][0]

  # Identify nodes and their depths
  def get_node_depths(adj: Array, root: int) -> Array:
    depths = jnp.full((n_nodes,), -1)
    depths = depths.at[root].set(0)

    # Simplified depth calculation: just iterate
    def step(_: int, d: Array) -> Array:
      # For each node j, if parent p is set, d[j] = d[p] + 1
      parents = jnp.argmax(adj, axis=1)
      # Avoid parent indexing if parent is itself or root etc?
      # In trees, every node except root has a unique parent.
      return jnp.where((d[parents] != -1) & (d == -1), d[parents] + 1, d)

    return jax.lax.fori_loop(0, n_nodes, step, depths)

  node_depths = get_node_depths(adjacency, root_node)
  max_depth = jnp.max(node_depths)

  # Identify Leaves
  # A node is a leaf if it is not a parent to any other node
  is_parent = jnp.any(
    adjacency == 1, axis=0
  )  # Correct: adjacency[child, parent] = 1. So j is a parent if any adjacency[:, j] == 1.
  leaves = jnp.where(~is_parent, size=n_nodes, fill_value=-1)[0]
  num_leaves = jnp.sum(leaves != -1)

  # 2. Key Generation
  # One key per node (representing the edge leading TO that node)
  key, subkey = jax.random.split(key)
  node_keys = jax.random.split(subkey, n_nodes)

  # 3. Path Construction for Leaves
  def get_leaf_path(leaf_idx: int) -> Array:
    # Trace back from leaf to root
    path = jnp.full((max_depth,), -1)

    def trace(i: int, state: tuple[int, Array]) -> tuple[int, Array]:
      curr, p_path = state
      parent = parent_indices[curr]
      # Only record if not at root
      p_path = jax.lax.cond(
        curr != root_node, lambda: p_path.at[max_depth - 1 - i].set(curr), lambda: p_path
      )
      return parent, p_path

    _, full_path = jax.lax.fori_loop(0, max_depth, trace, (leaf_idx, path))

    # Shift path to front
    valid_count = jnp.sum(full_path != -1)
    return jnp.roll(full_path, -(max_depth - valid_count))

  leaf_indices = leaves[:num_leaves]
  all_leaf_paths = jax.vmap(get_leaf_path)(leaf_indices)  # (num_leaves, max_depth)

  # Expand paths for branch_length
  # Each node index in the path is repeated branch_length times
  def expand_path(p: Array) -> Array:
    # Repeat each element. p has -1 for padding.
    return jnp.repeat(p, branch_length)

  expanded_paths = jax.vmap(expand_path)(all_leaf_paths)  # (num_leaves, max_depth * branch_length)

  # 4. Parallel Simulation
  initial_pop = jnp.repeat(root_sequence[None, :], pop_size, axis=0)

  # Use safe_map to process leaf paths in batches to conserve memory
  # Pass mutation_rate as dynamic argument to avoid recompilation
  mut_rates = jnp.full((expanded_paths.shape[0],), mutation_rate)

  all_lineages = safe_map(
    lambda inputs: _run_smc_path_with_traceback(
      path=inputs[0],
      initial_pop=initial_pop,
      node_keys=node_keys,
      landscape=landscape,
      mutation_rate=inputs[1],
      n_states=n_states,
      selection_intensity=selection_intensity,
      inference_batch_size=inference_batch_size,
      key=key,
    ),
    (expanded_paths, mut_rates),
    batch_size=inference_batch_size,
  )

  # 5. Aggregation
  # Initialize all_sequences with root
  all_sequences = jnp.zeros((n_nodes, seq_len), dtype=jnp.int32)
  all_sequences = all_sequences.at[root_node].set(root_sequence)

  # For each internal/leaf node, find one path that contains it and extract the sequence
  # Sequence for node 'n' is at the 'last' step of its edge in the path.
  def extract_node_sequence(node_idx: int) -> Array:
    # Find which leaf path contains node_idx.
    matches = all_leaf_paths == node_idx
    path_pos = jnp.where(matches, size=1)
    leaf_idx_in_vmap = path_pos[0][0]
    node_idx_in_path = path_pos[1][0]

    # The steps for this node in expanded_paths are:
    # node_idx_in_path * branch_length to (node_idx_in_path + 1) * branch_length - 1
    last_step_idx = (node_idx_in_path + 1) * branch_length - 1

    # Extract lineage sequence
    return all_lineages[leaf_idx_in_vmap, last_step_idx]

  # Nodes other than root
  other_nodes = jnp.where(jnp.arange(n_nodes) != root_node, size=n_nodes - 1)[0]

  def fill_sequences(i: int, seqs: Array) -> Array:
    node_idx = other_nodes[i]
    return seqs.at[node_idx].set(extract_node_sequence(node_idx))

  all_sequences = jax.lax.fori_loop(0, n_nodes - 1, fill_sequences, all_sequences)

  return nk_model.PhylogeneticTree(
    masked_sequences=jnp.zeros_like(all_sequences, dtype=nk_model.get_default_dtype()),
    all_sequences=all_sequences.astype(nk_model.get_default_dtype()),
    adjacency=adjacency.astype(nk_model.get_default_dtype()),
  )
