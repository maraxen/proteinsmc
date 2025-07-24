import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from esm.models.esmc import ESMC
import esmj
import chex
from proteinsmc.scoring.esm import make_esm_pll_score
from proteinsmc.utils.esm import remap_sequences
from proteinsmc.utils.constants import (
    COLABDESIGN_TO_ESM_AA_MAP_JAX,
    ESM_BOS_ID,
    ESM_EOS_ID,
    ESM_PAD_ID,
    ESM_MASK_ID, # Assuming ESM_MASK_ID is defined in your constants
)

@pytest.fixture(scope="module")
def esm_model():
    """
    Pytest fixture to load the ESM model once for all tests in this module.
    This avoids downloading and loading the model repeatedly.
    """
    # Using the smallest model for faster testing
    model_name = "esmc_300m"
    client = ESMC.from_pretrained(model_name)
    eqx_model = esmj.from_torch(client)
    return eqx.filter_jit(eqx_model)

def test_make_esm_pll_score_factory(esm_model):
    """
    Tests that the factory function returns a valid, callable scoring function.
    """
    # We pass the loaded model to avoid re-loading
    score_fn = make_esm_pll_score(model_name="esmc_300m", pll_method="per_position")
    assert callable(score_fn), "The factory did not return a callable function."

@pytest.mark.slow  # Mark this as a slow test
def test_esm_pll_score_vs_iterative_pll_multiple_sequences(esm_model):
  """
  Compares the O(1) PLL proxy score against a manually calculated,
  iterative single-position-masking PLL score for multiple input sequences.

  This test does NOT use a mock and relies on the actual model outputs
  to ensure the mathematical equivalence holds in practice.

  Args:
    esm_model: The ESM model loaded via the pytest fixture.

  Returns:
    None

  Raises:
    AssertionError: If the scores for any sequence do not match within tolerance.

  Example:
    >>> test_esm_pll_score_vs_iterative_pll_multiple_sequences(esm_model)
  """
  per_pos_score_fn = make_esm_pll_score(model_name="esmc_300m", pll_method="per_position")
  bayes_score_fn = make_esm_pll_score(model_name="esmc_300m", pll_method="bayes")

  key = jax.random.PRNGKey(0)

  sequences = jax.random.randint(
    key, (5, 200), 0, 20, dtype=jnp.int8
  )
  
  @jax.jit
  def get_scores(sequence):
    """Helper function to get scores for a single sequence."""
    per_pos_score = per_pos_score_fn(sequence, None, None)
    bayes_score = bayes_score_fn(sequence, None, None)
    return per_pos_score, bayes_score

  scores = jax.vmap(get_scores)(sequences)
  per_pos_scores, bayes_scores = scores

  assert jnp.allclose(per_pos_scores, bayes_scores, rtol=1e-2, atol=1e-2), (
      f"Scores do not align: proxy={bayes_scores}, iterative={per_pos_scores}"
    )
