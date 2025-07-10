"""Enhanced NK landscape tests using Chex testing framework."""

import jax
import jax.numpy as jnp
from jax import random as jax_random
import chex
import numpy.testing as npt

from proteinsmc.sampling.nk_landscape import (
    NKLandscape,
    generate_nk_interactions,
    generate_nk_model,
    calculate_nk_fitness_single,
    calculate_nk_fitness_population,
)


class TestNKLandscapeChex(chex.TestCase):
    """Test class using Chex's testing framework with variants."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.key = jax_random.PRNGKey(42)

    @chex.all_variants
    def test_generate_interactions_k0(self):
        """Test interaction generation for the K=0 edge case."""
        n, k = 5, 0
        
        @self.variant
        def generate_interactions(key):
            return generate_nk_interactions(key, n, k)
        
        interactions = generate_interactions(self.key)
        
        # Use Chex assertions
        chex.assert_shape(interactions, (n, k))
        chex.assert_equal(interactions.size, 0)

    @chex.all_variants
    def test_generate_interactions_properties(self):
        """Test properties of generated interactions for K > 0."""
        n, k = 10, 4
        
        @self.variant
        def generate_interactions(key):
            return generate_nk_interactions(key, n, k)
        
        interactions = generate_interactions(self.key)

        # Check shape using Chex
        chex.assert_shape(interactions, (n, k))

        # Check value range and finiteness
        chex.assert_tree_all_finite(interactions)
        assert jnp.all(interactions < n)
        assert jnp.all(interactions >= -1)

        # Check for self-interaction and uniqueness
        for i in range(n):
            assert i not in interactions[i]
            row_neighbors = interactions[i]
            valid_neighbors = row_neighbors[row_neighbors != -1]
            assert len(valid_neighbors) == len(jnp.unique(valid_neighbors))

    @chex.all_variants
    def test_generate_interactions_padding(self):
        """Test that interactions are padded with -1 if K > N-1."""
        n, k = 4, 5
        
        @self.variant
        def generate_interactions(key):
            return generate_nk_interactions(key, n, k)
        
        interactions = generate_interactions(self.key)

        chex.assert_shape(interactions, (n, k))
        for i in range(n):
            row = interactions[i]
            assert jnp.sum(row != -1) == n - 1
            assert jnp.sum(row == -1) == k - (n - 1)

    def test_generate_interactions_determinism(self):
        """Test that the same key produces the same interaction map."""
        n, k = 8, 3
        key = jax_random.PRNGKey(101)
        interactions1 = generate_nk_interactions(key, n, k)
        interactions2 = generate_nk_interactions(key, n, k)
        chex.assert_trees_all_equal(interactions1, interactions2)

    @chex.all_variants
    def test_generate_nk_model_structure(self):
        """Test the structure of the generated NKLandscape object."""
        n, k, q = 5, 2, 3
        
        @self.variant
        def generate_model(key):
            return generate_nk_model(key, n, k, q)
        
        landscape = generate_model(self.key)

        # Check type and structure
        assert isinstance(landscape, NKLandscape)
        chex.assert_shape(landscape.interactions, (n, k))
        
        expected_ft_shape = (n,) + (q,) * (k + 1)
        chex.assert_shape(landscape.fitness_tables, expected_ft_shape)

        # Check fitness table values are in [0, 1]
        chex.assert_tree_all_finite(landscape.fitness_tables)
        assert jnp.all(landscape.fitness_tables >= 0.0)
        assert jnp.all(landscape.fitness_tables <= 1.0)

    @chex.all_variants
    def test_calculate_fitness_single_k0(self):
        """Test fitness calculation for a single sequence with K=0."""
        n, k, q = 3, 0, 2
        
        # Create landscape
        interactions = jnp.full((n, k), -1, dtype=jnp.int32)
        fitness_tables = jnp.array([
            [0.1, 0.9],
            [0.2, 0.8],
            [0.3, 0.7],
        ]).reshape(n, q)
        landscape = NKLandscape(interactions, fitness_tables)
        
        sequence = jnp.array([1, 0, 1], dtype=jnp.int32)
        expected_fitness = jnp.mean(jnp.array([0.9, 0.2, 0.7]))

        @self.variant
        def calculate_fitness(seq, land):
            return calculate_nk_fitness_single(seq, land, n, k)
        
        actual_fitness = calculate_fitness(sequence, landscape)
        chex.assert_trees_all_close(actual_fitness, expected_fitness, atol=1e-6)

    @chex.all_variants
    def test_calculate_fitness_single_k_gt_0(self):
        """Test fitness calculation for a single sequence with K > 0."""
        n, k, q = 3, 1, 2

        interactions = jnp.array([[1], [2], [0]], dtype=jnp.int32)
        ft = jnp.arange(12, dtype=jnp.float32).reshape(3, 2, 2) / 12.0
        landscape = NKLandscape(interactions, ft)

        sequence = jnp.array([0, 1, 1], dtype=jnp.int32)

        # Calculate expected contributions
        contrib_0 = ft[0, 0, 1]
        contrib_1 = ft[1, 1, 1]
        contrib_2 = ft[2, 1, 0]
        expected_fitness = jnp.mean(jnp.array([contrib_0, contrib_1, contrib_2]))

        @self.variant
        def calculate_fitness(seq, land):
            return calculate_nk_fitness_single(seq, land, n, k)
        
        actual_fitness = calculate_fitness(sequence, landscape)
        chex.assert_trees_all_close(actual_fitness, expected_fitness, atol=1e-6)

    @chex.all_variants
    def test_calculate_fitness_population(self):
        """Test vmapped fitness calculation."""
        n, k, q = 3, 1, 2

        interactions = jnp.array([[1], [2], [0]], dtype=jnp.int32)
        ft = jnp.arange(12, dtype=jnp.float32).reshape(3, 2, 2) / 12.0
        landscape = NKLandscape(interactions, ft)

        population = jnp.array([
            [0, 1, 1],
            [1, 0, 1],
        ], dtype=jnp.int32)

        expected_fitnesses = jnp.array([0.5, 0.5])

        @self.variant
        def calculate_pop_fitness(pop, land):
            return calculate_nk_fitness_population(pop, land, n, k)
        
        actual_fitnesses = calculate_pop_fitness(population, landscape)
        
        chex.assert_shape(actual_fitnesses, (2,))
        chex.assert_trees_all_close(actual_fitnesses, expected_fitnesses, atol=1e-6)

    def test_jit_compatibility(self):
        """Test that functions work with and without JIT compilation."""
        n, k, q = 4, 2, 3
        
        # Test without JIT
        def full_pipeline_no_jit(key):
            landscape = generate_nk_model(key, n, k, q)
            sequence = jnp.array([0, 1, 2, 1], dtype=jnp.int32)
            return calculate_nk_fitness_single(sequence, landscape, n, k)
        
        # Test with JIT
        @jax.jit
        def full_pipeline_jit(key):
            landscape = generate_nk_model(key, n, k, q)
            sequence = jnp.array([0, 1, 2, 1], dtype=jnp.int32)
            return calculate_nk_fitness_single(sequence, landscape, n, k)
        
        fitness_no_jit = full_pipeline_no_jit(self.key)
        fitness_jit = full_pipeline_jit(self.key)
        
        # Check that both are finite and in reasonable range
        chex.assert_tree_all_finite(fitness_no_jit)
        chex.assert_tree_all_finite(fitness_jit)
        assert 0.0 <= fitness_no_jit <= 1.0
        assert 0.0 <= fitness_jit <= 1.0
        
        # Results should be the same
        chex.assert_trees_all_close(fitness_no_jit, fitness_jit, atol=1e-6)

    def test_dataclass_pytree_integration(self):
        """Test that NKLandscape works properly as a PyTree with dataclass."""
        n, k, q = 3, 1, 2
        landscape = generate_nk_model(self.key, n, k, q)
        
        # Test tree_map operations
        scaled_landscape = jax.tree_util.tree_map(
            lambda x: x * 2.0 if jnp.issubdtype(x.dtype, jnp.floating) else x, landscape
        )
        
        assert isinstance(scaled_landscape, NKLandscape)
        chex.assert_shape(scaled_landscape.interactions, landscape.interactions.shape)
        chex.assert_shape(scaled_landscape.fitness_tables, landscape.fitness_tables.shape)
        
        # Verify scaling worked on fitness tables but not interactions
        chex.assert_trees_all_equal(scaled_landscape.interactions, landscape.interactions)
        chex.assert_trees_all_close(
            scaled_landscape.fitness_tables, 
            landscape.fitness_tables * 2.0, 
            atol=1e-6
        )
