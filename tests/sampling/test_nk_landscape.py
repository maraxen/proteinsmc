import jax
import jax.numpy as jnp
import pytest
from proteinsmc.sampling.nk_landscape import (
    generate_nk_interactions,
    generate_nk_model,
    calculate_nk_fitness_single,
    calculate_nk_fitness_population,
)
from proteinsmc.models.nk_landscape import NKLandscape

class TestNKLandscape:
    """Tests for the NK landscape generation and fitness calculation."""

    @pytest.mark.parametrize("n, k", [(10, 2), (20, 5), (5, 0), (5, 4)])
    def test_generate_nk_interactions(self, n, k):
        """Tests the generation of the NK interaction map."""
        key = jax.random.PRNGKey(0)
        interactions = generate_nk_interactions(key, n, k)

        assert interactions.shape == (n, k)

        # Check that for each site, its own index is not in its neighbors
        for i in range(n):
            assert i not in interactions[i]

        # Check that neighbors are unique for each site
        for i in range(n):
            # Filter out padding
            valid_neighbors = interactions[i][interactions[i] != -1]
            assert len(valid_neighbors) == len(jnp.unique(valid_neighbors))

    @pytest.mark.parametrize("n, k, q", [(10, 2, 2), (5, 3, 4)])
    def test_generate_nk_model(self, n, k, q):
        """Tests the generation of the full NK model."""
        key = jax.random.PRNGKey(0)
        landscape = generate_nk_model(key, n, k, q)

        assert isinstance(landscape, NKLandscape)
        assert landscape.interactions.shape == (n, k)
        expected_fitness_shape = (n,) + (q,) * (k + 1)
        assert landscape.fitness_tables.shape == expected_fitness_shape

    def test_deterministic_seeding(self):
        """Tests that the same seed produces the same landscape."""
        key = jax.random.PRNGKey(42)
        n, k, q = 10, 3, 2

        landscape1 = generate_nk_model(key, n, k, q)
        landscape2 = generate_nk_model(key, n, k, q)

        assert jnp.array_equal(landscape1.interactions, landscape2.interactions)
        assert jnp.array_equal(landscape1.fitness_tables, landscape2.fitness_tables)

    @pytest.mark.parametrize("n, k, q", [(10, 2, 2), (8, 4, 3)])
    def test_calculate_nk_fitness_single(self, n, k, q):
        """Tests the fitness calculation for a single sequence."""
        key = jax.random.PRNGKey(0)
        landscape = generate_nk_model(key, n, k, q)
        sequence = jax.random.randint(key, (n,), 0, q)

        fitness = calculate_nk_fitness_single(sequence, landscape, n, k)
        assert isinstance(fitness, jnp.ndarray)
        assert fitness.shape == ()

    @pytest.mark.parametrize("n, k, q, pop_size", [(10, 2, 2, 8), (8, 4, 3, 4)])
    def test_calculate_nk_fitness_population(self, n, k, q, pop_size):
        """Tests the fitness calculation for a population of sequences."""
        key = jax.random.PRNGKey(0)
        landscape = generate_nk_model(key, n, k, q)
        population = jax.random.randint(key, (pop_size, n), 0, q)

        fitness_scores = calculate_nk_fitness_population(population, landscape, n, k)
        assert isinstance(fitness_scores, jnp.ndarray)
        assert fitness_scores.shape == (pop_size,)

    def test_k0_special_case(self):
        """Tests the K=0 edge case."""
        n, k, q = 10, 0, 2
        key = jax.random.PRNGKey(0)
        landscape = generate_nk_model(key, n, k, q)
        sequence = jax.random.randint(key, (n,), 0, q)

        fitness = calculate_nk_fitness_single(sequence, landscape, n, k)

        # For K=0, fitness is the mean of individual site contributions
        expected_fitness = jnp.mean(landscape.fitness_tables[jnp.arange(n), sequence])
        assert jnp.allclose(fitness, expected_fitness)