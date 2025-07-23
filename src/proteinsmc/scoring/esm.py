"""ESM-based protein sequence scoring function."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import equinox as eqx
import esmj
import jax
import jax.numpy as jnp
from esm.models.esmc import ESMC

from proteinsmc.utils.constants import ESM_MASK_ID
from proteinsmc.utils.esm import remap_sequences

if TYPE_CHECKING:
  from jaxtyping import Array, Float, Int, PRNGKeyArray

  from proteinsmc.models.fitness import FitnessFn
  from proteinsmc.models.types import ProteinSequence


PLL_ALPHA = 0.15
PLL_BETA = 0.85


def make_esm_pll_score(
  model_name: Literal["esmc_300m", "esmc_600m"],
  pll_method: Literal["ofs", "per_position", "bayes"],
) -> FitnessFn:
  """Create an ESM-based protein sequence scoring function.

  This function loads a specified ESM-J model and returns a scoring function
  that calculates the pseudo-log-likelihood (PLL) of a protein sequence.

  Args:
      model_name: The name of the ESM model to load.

  Returns:
      A FitnessFn that takes a protein sequence and returns its PLL score.

  """
  client = ESMC.from_pretrained(model_name)
  eqx_model = esmj.from_torch(client)
  eqx_model = eqx.filter_jit(eqx_model)

  match pll_method:
    case "ofs":
      msg = "The 'ofs' PLL method is not implemented yet."
      raise NotImplementedError(msg)
    case "per_position":

      def esm_pll_score(
        sequence: ProteinSequence,
        _key: PRNGKeyArray | None = None,
        _context: Array | None = None,
      ) -> Float:
        """Calculate the O(1) PLL proxy for a protein sequence using an ESM model.

        This method uses the per-position PLL calculation.

        Args:
            _key: JAX PRNG key (unused).
            sequence: The protein sequence to score.
            _context: Additional context (unused).

        Returns:
            The PLL score for the protein sequence.

        """
        sequence = sequence.astype(jnp.int16)
        original_esm_ids, _ = remap_sequences(sequence)

        def pll_for_position(i: Int) -> Float:
          # Mask the i-th position
          masked_sequence = sequence.at[i].set(ESM_MASK_ID)
          tokens, _ = remap_sequences(masked_sequence)
          logits = eqx_model(tokens[None]).logits[0, 1:-1, :]
          log_probs_at_position = jax.nn.log_softmax(logits[i])
          original_token_id = original_esm_ids[i + 1]  # +1 for BOS token
          return log_probs_at_position[original_token_id]

        # Vectorize over all positions
        log_probs = jax.vmap(pll_for_position)(jnp.arange(sequence.shape[0]))
        total_log_prob = jnp.sum(log_probs)

        return total_log_prob / sequence.shape[0]
    case "bayes":

      def esm_pll_score(
        sequence: ProteinSequence,
        _key: PRNGKeyArray | None = None,
        _context: Array | None = None,
      ) -> Float:
        """Calculate the O(1) PLL proxy for a protein sequence using an ESM model.

        This method leverages Bayes' theorem to correct the output logits,
        accounting for the model's masking strategy during training.

        Args:
            _key: JAX PRNG key (unused).
            sequence: The protein sequence to score.
            _context: Additional context (unused).

        Returns:
            The PLL score for the protein sequence.

        """
        sequence = sequence.astype(jnp.int16)
        tokens, _ = remap_sequences(sequence)
        logits = eqx_model(tokens[None]).logits[0, 1:-1, :]
        log_likelihood = jax.nn.log_softmax(logits)
        e_dist = jnp.ones(logits.shape[-1], dtype=jnp.float32) / logits.shape[-1]
        factor1 = (PLL_ALPHA + PLL_BETA) / PLL_ALPHA
        factor2 = PLL_BETA / PLL_ALPHA
        p_prime = jnp.maximum((factor1 * log_likelihood) - (factor2 * e_dist), 1e-6)
        row_sums = jnp.sum(p_prime, axis=-1, keepdims=True)
        row_sums = jnp.where(row_sums < 1e-6,
                              jnp.ones_like(row_sums),
                              row_sums)
        p_prime_normalized = p_prime / row_sums
        

        pll = (log_likelihood - jnp.log(PLL_ALPHA * jnp.exp(log_likelihood) + PLL_BETA)) / (
          1 - PLL_ALPHA
        )
        return jnp.mean(pll) / sequence.shape[0]
    case _:
      msg = f"Unknown PLL method: {pll_method}"
      raise ValueError(msg)
  return esm_pll_score


  """ Calculates the O(1) PLL proxy based on Algorithm 2. """
    if logits is None or sequence_string is None:
        return 0.0
    L, A = logits.shape
    if A != ALPHABET_SIZE or L != len(sequence_string):
        return 0.0
    alpha, beta, epsilon = PLL_ALPHA, PLL_BETA, PLL_EPSILON
    if alpha <= 0:
        raise ValueError("PLL_ALPHA must be positive.")
    p = softmax(logits.astype(np.float64), axis=-1)
    e_dist = np.ones(A, dtype=np.float64) / A
    factor1 = (alpha + beta) / alpha
    factor2 = beta / alpha
    p_prime = np.maximum((factor1 * p) - (factor2 * e_dist[np.newaxis, :]), epsilon)
    row_sums = np.sum(p_prime, axis=-1, keepdims=True)
    row_sums[row_sums < epsilon] = 1.0 # Avoid division by near zero
    p_prime_normalized = p_prime / row_sums
    pll = 0.0
    valid_len = 0
    for i in range(L):
        native_aa = sequence_string[i]
        if native_aa in AA_TO_IDX:
            native_idx_0_19 = AA_TO_IDX[native_aa]
            prob_native = p_prime_normalized[i, native_idx_0_19]
            pll += np.log(np.maximum(prob_native, epsilon)) # Use epsilon floor
            valid_len += 1
        else:
            warnings.warn(f"Non-standard AA '{native_aa}' at pos {i}.")
    return pll / valid_len if valid_len > 0 else 0.0