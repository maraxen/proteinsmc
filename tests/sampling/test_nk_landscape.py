import jax
import jax.numpy as jnp
import pytest
from jax import random

from proteinsmc.sampling.nk_landscape import (
  calculate_nk_fitness_population_jax,
  calculate_nk_fitness_single_jax,
  generate_nk_interactions_jax,
  get_nk_site_contribution_jax,
)


@pytest.mark.parametrize("N, K, q", [(10, 2, 4), (20, 4, 2), (5, 0, 3), (5, 5, 2)])
def test_generate_nk_interactions_jax(N, K, q):
  key = random.PRNGKey(0)
  interactions = generate_nk_interactions_jax(key, N, K, q)

  assert interactions.shape == (N, K)
  assert interactions.dtype == jnp.int32

  # Check that no site is its own neighbor
  for i in range(N):
    assert i not in interactions[i, :]

  # Check padding for K >= N-1
  if K >= N - 1 and N > 1:
    num_valid_neighbors = N - 1
    assert jnp.all(interactions[:, num_valid_neighbors:] == -1)


def test_generate_nk_interactions_jax_edge_cases():
  key = random.PRNGKey(0)
  # Test N=1
  interactions_n1 = generate_nk_interactions_jax(key, 1, 0, 2)
  assert interactions_n1.shape == (1, 0)

  # Test K=0
  interactions_k0 = generate_nk_interactions_jax(key, 10, 0, 2)
  assert interactions_k0.shape == (10, 0)


@pytest.mark.parametrize("K_plus_1", [3, 5])
def test_get_nk_site_contribution_jax(K_plus_1):
  key = random.PRNGKey(42)
  site_idx = 1
  site_state = 0
  K = K_plus_1 - 1
  neighbor_states = jnp.arange(K)

  contribution1 = get_nk_site_contribution_jax(site_idx, site_state, neighbor_states, key, K_plus_1)
  contribution2 = get_nk_site_contribution_jax(site_idx, site_state, neighbor_states, key, K_plus_1)

  assert jnp.isclose(contribution1, contribution2)
  assert 0.0 <= contribution1 <= 1.0

  # Different key should produce different result
  key2 = random.PRNGKey(43)
  contribution3 = get_nk_site_contribution_jax(
    site_idx, site_state, neighbor_states, key2, K_plus_1
  )
  assert not jnp.isclose(contribution1, contribution3)


@pytest.mark.parametrize("N, K, q", [(5, 2, 2), (10, 3, 4)])
def test_calculate_nk_fitness_single_jax(N, K, q):
  key = random.PRNGKey(0)
  config = random.randint(key, (N,), 0, q)
  interactions_map = generate_nk_interactions_jax(key, N, K, q)
  landscape_master_key = random.PRNGKey(1)

  fitness = calculate_nk_fitness_single_jax(config, interactions_map, landscape_master_key, N, K, q)

  assert isinstance(fitness, jax.Array)
  assert fitness.shape == ()
  assert 0.0 <= fitness <= 1.0


def test_calculate_nk_fitness_population_jax(N=10, K=2, q=2, population_size=8):
  key = random.PRNGKey(0)
  key_pop, key_landscape = random.split(key)

  configs_population = random.randint(key_pop, (population_size, N), 0, q)
  interactions_map = generate_nk_interactions_jax(key_landscape, N, K, q)
  landscape_master_key = random.PRNGKey(1)

  # Calculate population fitness
  pop_fitness = calculate_nk_fitness_population_jax(
    configs_population, interactions_map, landscape_master_key, N, K, q
  )

  assert pop_fitness.shape == (population_size,)

  # Calculate fitness for each individual and compare
  individual_fitnesses = []
  for i in range(population_size):
    fitness = calculate_nk_fitness_single_jax(
      configs_population[i], interactions_map, landscape_master_key, N, K, q
    )
    individual_fitnesses.append(fitness)

  assert jnp.allclose(pop_fitness, jnp.array(individual_fitnesses))
