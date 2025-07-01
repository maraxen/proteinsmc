from functools import partial

import jax.numpy as jnp
from jax import jit, random
from jaxtyping import PRNGKeyArray

from .types import (
    BatchSequenceFloats,
    BatchSequences,
)


@partial(jit, static_argnames=('n_particles',))
def resampling_kernel(
    key: PRNGKeyArray,
    sequences: BatchSequences,
    log_weights: BatchSequenceFloats,
    n_particles: int
) -> tuple[BatchSequences, BatchSequenceFloats, BatchSequenceFloats]:
    """Performs multinomial resampling based on log_weights."""
    # Stabilize log_weights to prevent overflow/underflow when exponentiating
    max_log_weight = jnp.max(
        jnp.where(jnp.isneginf(log_weights), -jnp.inf, log_weights)
    )
    # Handle case where all log_weights are -inf: set max_log_weight to 0.0 to
    # avoid NaN
    safe_max_log_weight = jnp.where(
        jnp.isneginf(max_log_weight) & jnp.all(jnp.isneginf(log_weights)),
        0.0, max_log_weight
    )

    stable_log_weights = log_weights - safe_max_log_weight
    weights_unnormalized = jnp.exp(stable_log_weights)
    # Ensure particles with original -inf log_weight get 0 probability
    weights_unnormalized = jnp.where(
        jnp.isneginf(log_weights) | jnp.isinf(log_weights) |
        jnp.isnan(log_weights), 0.0, weights_unnormalized
    )

    sum_weights = jnp.sum(weights_unnormalized)

    # If sum_weights is invalid (e.g., zero or NaN), use uniform probabilities
    uniform_probs = jnp.ones(n_particles, dtype=jnp.float32) / n_particles
    is_sum_weights_ok = (sum_weights > 1e-9) & jnp.isfinite(sum_weights)

    normalized_weights = jnp.where(
        is_sum_weights_ok,
        weights_unnormalized / sum_weights,
        uniform_probs
    )
    # Clean up potential NaNs if uniform_probs was used due to bad sum_weights
    normalized_weights = jnp.where(
        jnp.isnan(normalized_weights), 0.0, normalized_weights
    )
    # If all weights became 0 (e.g. after NaN handling), re-uniformize to
    # prevent choice error
    normalized_weights = jnp.where(
        jnp.sum(normalized_weights) < 1e-9, uniform_probs, normalized_weights
    )

    # Calculate Effective Sample Size (ESS)
    ess = 1.0 / jnp.sum(normalized_weights**2)
    ess = jnp.where(is_sum_weights_ok, ess, 0.0) # ESS is 0 if weights were problematic

    # Perform resampling: draw particle indices with replacement
    indices = random.choice(
        key, jnp.arange(n_particles), shape=(n_particles,),
        p=normalized_weights, replace=True
    )
    resampled_sequences = sequences[indices]
    return resampled_sequences, ess, normalized_weights

