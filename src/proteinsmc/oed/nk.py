"""NK landscape experimental design module."""

from jaxtyping import PRNGKeyArray

from proteinsmc.models.nk_landscape import NKLandscape
from proteinsmc.oed.structs import OEDDesign
from proteinsmc.utils.nk_landscape import generate_nk_model


def create_landscape_from_design(key: PRNGKeyArray, design: OEDDesign) -> NKLandscape:
  """Create an NK landscape based on OED design parameters.

  Args:
      key: JAX PRNG key.
      design: OED design parameters.

  Returns:
      NKLandscape object configured according to design parameters.

  Example:
      >>> design = OEDDesign(N=20, K=3, q=4, population_size=100,
      ...                   n_generations=50, mutation_rate=0.01,
      ...                   diversification_ratio=0.1)
      >>> landscape = create_landscape_from_design(jax.random.PRNGKey(42), design)

  """
  return generate_nk_model(key, design.N, design.K, design.q)
