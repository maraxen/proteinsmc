import jax
import jax.numpy as jnp
from proteinsmc.oed.nk import create_landscape_from_design
from proteinsmc.oed.structs import OEDDesign


def test_create_landscape_shapes():
    design = OEDDesign(N=8, K=2, q=3, population_size=10, n_generations=4, mutation_rate=0.01, diversification_ratio=0.1)
    key = jax.random.PRNGKey(0)
    landscape = create_landscape_from_design(key, design)
    # interactions shape should be (N, K)
    assert landscape.interactions.shape == (design.N, design.K)
    # fitness_tables should have shape (N, q, ...)
    assert landscape.fitness_tables.shape[0] == design.N

