"""ESM-based protein sequence scoring function."""

from __future__ import annotations

from functools import partial
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


# PLL_ALPHA = 0.25
# PLL_BETA = 0.75
PLL_EPSILON = 1e-6


def make_esm_pll_score(
  model_name: Literal["esmc_300m", "esmc_600m"],
  pll_method: Literal["ofs", "per_position", "bayes"],
  method_kwargs: dict[str, float] | None = None,
) -> FitnessFn:
  """Create an ESM-based protein sequence scoring function.

  This function loads a specified ESM-J model and returns a scoring function
  that calculates the pseudo-log-likelihood (PLL) of a protein sequence.

  Args:
      model_name: The name of the ESM model to load.
      pll_method: The method to use for calculating the PLL score.

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

      @partial(jax.jit, static_argnames=("_key", "_context"))
      def esm_pll_score(
        sequence: ProteinSequence,
        _key: PRNGKeyArray | None = None,
        _context: Array | None = None,
      ) -> Float:
        """Calculate the standard per-position masking PLL score for a protein sequence.

        This method uses the per-position PLL calculation standard of masking each
        position in the sequence and computing the likelihood of the original token
        given the masked context.

        Args:
            _key: JAX PRNG key (unused).
            sequence: The protein sequence to score.
            _context: Additional context (unused).

        Returns:
            The PLL score for the protein sequence.

        """
        sequence = sequence.astype(jnp.int16)
        original_esm_ids, _ = remap_sequences(sequence)

        @jax.jit
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
      PLL_ALPHA = method_kwargs.get("alpha", 0.15)
      PLL_BETA = method_kwargs.get("beta", 0.85)

      @partial(jax.jit, static_argnames=("_key", "_context"))
      def esm_pll_score(
        sequence: ProteinSequence,
        _key: PRNGKeyArray | None = None,
        _context: Array | None = None,
      ) -> Float:
        """Calculate the O(1) PLL proxy for a protein sequence using an ESM model.

        This method leverages Theorem 4.1 from Gordon et al. 2025
        to correct the output probabilities, accounting for the model's masking strategy
        during training.

        Args:
            sequence: The protein sequence to score.
            _key: JAX PRNG key (unused).
            _context: Additional context (unused).

        Returns:
            The PLL score for the protein sequence.

        """
        sequence = sequence.astype(jnp.int16)
        tokens, _ = remap_sequences(sequence)
        logits = eqx_model(tokens[None]).logits[0, 1:-1, :]

        # Get the model's probabilities P(yi = xi | x, θ)
        model_probs = jax.nn.softmax(logits)

        # Apply Theorem 4.1: P(yi = xi | x\i, θ) = ((α + β)/α) P(yi = xi | x, θ) - (β/α)
        # where α = PLL_ALPHA and β = PLL_BETA
        uniform_prob = 1.0 / logits.shape[-1]
        corrected_probs = ((PLL_ALPHA + PLL_BETA) / PLL_ALPHA) * model_probs - (
          PLL_BETA / PLL_ALPHA
        ) * uniform_prob

        # Ensure probabilities are valid (clamp to epsilon for numerical stability)
        corrected_probs = jnp.maximum(corrected_probs, PLL_EPSILON)

        # Get the corrected probabilities for the actual sequence tokens
        idx = jnp.arange(sequence.shape[0])
        token_probs = corrected_probs[idx, sequence]

        # Calculate log-likelihood and normalize by sequence length
        log_probs = jnp.log(jnp.maximum(token_probs, PLL_EPSILON))
        total_log_prob = jnp.sum(log_probs)

        return total_log_prob / sequence.shape[0]
    case _:
      msg = f"Unknown PLL method: {pll_method}"
      raise ValueError(msg)
  return esm_pll_score
