import pytest
import jax
import jax.numpy as jnp
from jax import random as jax_random
import numpy.testing as npt

from proteinsmc.sampling.nk_landscape import (
    NKLandscape,
    generate_nk_interactions,
    generate_nk_model,
    calculate_nk_fitness_single_jax,
    calculate_nk_fitness_population_jax,
)

@pytest.fixture
def key():
    """Provides a base JAX PRNG key for all tests."""
    return jax_random.PRNGKey(42)


def test_generate_interactions_k0(key):
    """Test interaction generation for the K=0 edge case."""
    n, k = 5, 0
    interactions = generate_nk_interactions(key, n, k)
    assert interactions.shape == (n, k)
    # The array should be empty but have the correct shape.
    assert interactions.size == 0


def test_generate_interactions_properties(key):
    """Test properties of generated interactions for K > 0."""
    n, k = 10, 4
    interactions = generate_nk_interactions(key, n, k)

    # 1. Check shape
    assert interactions.shape == (n, k)

    # 2. Check value range (indices must be valid or -1 for padding)
    assert jnp.all(interactions < n)
    assert jnp.all(interactions >= -1)

    # 3. Check for self-interaction (a site should not be its own neighbor)
    for i in range(n):
        assert i not in interactions[i]

    # 4. Check for unique neighbors in each row
    for i in range(n):
        row_neighbors = interactions[i]
        # Filter out padding
        valid_neighbors = row_neighbors[row_neighbors != -1]
        assert len(valid_neighbors) == len(jnp.unique(valid_neighbors))


def test_generate_interactions_padding(key):
    """Test that interactions are padded with -1 if K > N-1."""
    n, k = 4, 5  # K is greater than the number of possible neighbors (N-1 = 3)
    interactions = generate_nk_interactions(key, n, k)

    assert interactions.shape == (n, k)
    for i in range(n):
        row = interactions[i]
        # There should be N-1 valid neighbors
        assert jnp.sum(row != -1) == n - 1
        # The rest should be padding
        assert jnp.sum(row == -1) == k - (n - 1)


def test_generate_interactions_determinism():
    """Test that the same key produces the same interaction map."""
    n, k = 8, 3
    key = jax_random.PRNGKey(101)
    interactions1 = generate_nk_interactions(key, n, k)
    interactions2 = generate_nk_interactions(key, n, k)
    npt.assert_array_equal(interactions1, interactions2)


def test_generate_nk_model_structure(key):
    """Test the structure of the generated NKLandscape object."""
    n, k, q = 5, 2, 3
    landscape = generate_nk_model(key, n, k, q)

    # 1. Check type
    assert isinstance(landscape, NKLandscape)

    # 2. Check interactions shape
    assert landscape.interactions.shape == (n, k)

    # 3. Check fitness tables shape
    expected_ft_shape = (n,) + (q,) * (k + 1)
    assert landscape.fitness_tables.shape == expected_ft_shape

    # 4. Check fitness table values (should be in [0, 1])
    assert jnp.all(landscape.fitness_tables >= 0.0)
    assert jnp.all(landscape.fitness_tables <= 1.0)


def test_calculate_fitness_single_k0():
    """Test fitness calculation for a single sequence with K=0."""
    n, k, q = 3, 0, 2
    
    # Manually create a simple landscape
    interactions = jnp.full((n, k), -1, dtype=jnp.int32)
    # Fitness depends only on the site's own state
    fitness_tables = jnp.array([
        [0.1, 0.9],  # Site 0: state 0 -> 0.1, state 1 -> 0.9
        [0.2, 0.8],  # Site 1: state 0 -> 0.2, state 1 -> 0.8
        [0.3, 0.7],  # Site 2: state 0 -> 0.3, state 1 -> 0.7
    ]).reshape(n, q) # Reshape to (N, q) for K=0

    landscape = NKLandscape(interactions, fitness_tables)
    
    # Sequence: [1, 0, 1]
    sequence = jnp.array([1, 0, 1], dtype=jnp.int32)
    
    # Expected contributions
    # Site 0, state 1 -> 0.9
    # Site 1, state 0 -> 0.2
    # Site 2, state 1 -> 0.7
    expected_contributions = jnp.array([0.9, 0.2, 0.7])
    expected_fitness = jnp.mean(expected_contributions) # (0.9 + 0.2 + 0.7) / 3

    # Calculate fitness using the JAX function
    actual_fitness = calculate_nk_fitness_single_jax(sequence, landscape, n, k)
    
    npt.assert_almost_equal(actual_fitness, expected_fitness, decimal=6)


def test_calculate_fitness_single_k_gt_0():
    """Test fitness calculation for a single sequence with K > 0."""
    n, k, q = 3, 1, 2

    # Manually create a simple landscape
    # Site 0 influenced by 1, 1 by 2, 2 by 0
    interactions = jnp.array([[1], [2], [0]], dtype=jnp.int32)
    
    # Fitness tables shape: (N, q, q) -> (3, 2, 2)
    # fitness_tables[site, state_of_site, state_of_neighbor]
    ft = jnp.arange(12, dtype=jnp.float32).reshape(3, 2, 2) / 12.0
    landscape = NKLandscape(interactions, ft)

    # Sequence: [0, 1, 1]
    sequence = jnp.array([0, 1, 1], dtype=jnp.int32)

    # Expected contributions:
    # Site 0 (state 0), neighbor is site 1 (state 1) -> ft[0, 0, 1]
    contrib_0 = ft[0, 0, 1]
    # Site 1 (state 1), neighbor is site 2 (state 1) -> ft[1, 1, 1]
    contrib_1 = ft[1, 1, 1]
    # Site 2 (state 1), neighbor is site 0 (state 0) -> ft[2, 1, 0]
    contrib_2 = ft[2, 1, 0]
    
    expected_fitness = jnp.mean(jnp.array([contrib_0, contrib_1, contrib_2]))

    actual_fitness = calculate_nk_fitness_single_jax(sequence, landscape, n, k)

    npt.assert_almost_equal(actual_fitness, expected_fitness, decimal=6)


def test_calculate_fitness_population(key):
    """Test vmapped fitness calculation for a population."""
    n, k, q = 4, 2, 2
    pop_size = 10

    # Generate a random landscape and population
    key_land, key_pop = jax_random.split(key)
    landscape = generate_nk_model(key_land, n, k, q)
    population = jax_random.randint(key_pop, shape=(pop_size, n), minval=0, maxval=q)

    # Method 1: Calculate using the vmapped population function
    pop_fitness_vmap = calculate_nk_fitness_population_jax(population, landscape, n, k)

    # Method 2: Calculate by looping and calling the single function
    pop_fitness_loop = jnp.array([
        calculate_nk_fitness_single_jax(ind, landscape, n, k) for ind in population
    ])

    assert pop_fitness_vmap.shape == (pop_size,)
    npt.assert_almost_equal(pop_fitness_vmap, pop_fitness_loop, decimal=6)
