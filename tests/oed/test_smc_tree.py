
"""Unit tests for SMC-based tree generation with redundant parallelism."""

import jax
import jax.numpy as jnp
import pytest
import numpy as np

from proteinsmc.oed.smc_tree import generate_tree_data_smc
from trex.nk_model import create_nk_model_landscape
from trex.evals.benchmark import create_balanced_binary_tree

class TestSMCTree:
    """Tests for SMC-based tree generation."""

    @pytest.fixture
    def setup_landscape_and_tree(self):
        """Setup a simple landscape and tree for testing."""
        key = jax.random.PRNGKey(42)
        n = 20
        k = 2
        q = 4
        n_leaves = 8
        landscape = create_nk_model_landscape(n, k, key, n_states=q)
        adj_matrix = create_balanced_binary_tree(n_leaves)
        
        key, subkey = jax.random.split(key)
        root_sequence = jax.random.randint(subkey, (n,), 0, q)
        
        return landscape, adj_matrix, root_sequence, q

    def test_basic_generation(self, setup_landscape_and_tree):
        """Test that generate_tree_data_smc produces a valid PhylogeneticTree."""
        landscape, adj_matrix, root_sequence, q = setup_landscape_and_tree
        key = jax.random.PRNGKey(123)
        mutation_rate = 0.05
        pop_size = 10
        
        tree = generate_tree_data_smc(
            landscape, adj_matrix, root_sequence, mutation_rate, 
            key, pop_size=pop_size, n_states=q
        )
        
        assert tree.all_sequences.shape == (adj_matrix.shape[0], root_sequence.shape[0])
        assert tree.adjacency.shape == adj_matrix.shape
        
        # Verify root sequence matches (approximately, as it might be float32)
        root_node = jnp.argmax(jnp.sum(adj_matrix, axis=0) == 0) # This is not robust for all trees but for balanced it is node 0 typically or last.
        # Better: find root using parent_indices logic from implementation
        parent_indices = jnp.argmax(adj_matrix, axis=1)
        n_nodes = adj_matrix.shape[0]
        root_idx = jnp.where(parent_indices == jnp.arange(n_nodes), size=1)[0][0]
        
        np.testing.assert_array_equal(tree.all_sequences[root_idx].astype(jnp.int32), root_sequence)

    def test_shared_history_determinism(self, setup_landscape_and_tree):
        """Verify that identical keys produce identical trees."""
        landscape, adj_matrix, root_sequence, q = setup_landscape_and_tree
        key = jax.random.PRNGKey(456)
        mutation_rate = 0.1
        pop_size = 5
        
        tree1 = generate_tree_data_smc(
            landscape, adj_matrix, root_sequence, mutation_rate, 
            key, pop_size=pop_size, n_states=q
        )
        
        tree2 = generate_tree_data_smc(
            landscape, adj_matrix, root_sequence, mutation_rate, 
            key, pop_size=pop_size, n_states=q
        )
        
        np.testing.assert_array_equal(tree1.all_sequences, tree2.all_sequences)
        
        # Different key should produce different result
        key_diff = jax.random.PRNGKey(789)
        tree3 = generate_tree_data_smc(
            landscape, adj_matrix, root_sequence, mutation_rate, 
            key_diff, pop_size=pop_size, n_states=q
        )
        
        assert not jnp.array_equal(tree1.all_sequences, tree3.all_sequences)

    def test_branch_length_effect(self, setup_landscape_and_tree):
        """Test that longer branch lengths produce more divergent sequences."""
        landscape, adj_matrix, root_sequence, q = setup_landscape_and_tree
        key = jax.random.PRNGKey(101)
        mutation_rate = 0.05
        pop_size = 20
        
        # Helper to get mean distance from root for all nodes
        def get_mean_dist(branch_len, key):
            tree = generate_tree_data_smc(
                landscape, adj_matrix, root_sequence, mutation_rate, 
                key, pop_size=pop_size, branch_length=branch_len, n_states=q
            )
            # Calculate Hamming distance from root for all nodes
            parent_indices = jnp.argmax(adj_matrix, axis=1)
            n_nodes = adj_matrix.shape[0]
            root_idx = jnp.where(parent_indices == jnp.arange(n_nodes), size=1)[0][0]
            
            root_seq_bc = root_sequence[None, :]
            all_seqs = tree.all_sequences.astype(jnp.int32)
            node_dists = jnp.mean(all_seqs != root_seq_bc, axis=1)
            # Exclude root
            non_root_dists = node_dists.at[root_idx].set(0.0)
            return jnp.sum(non_root_dists) / (n_nodes - 1)

        # 1. Branch length = 1
        dist_1 = get_mean_dist(1, key)
        
        # 2. Branch length = 5
        dist_5 = get_mean_dist(5, key)
        
        assert dist_5 > dist_1, f"Expected more divergence with length 5 vs 1. Got {dist_5} vs {dist_1}"

