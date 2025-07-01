"""
This module implements the Gibbs sampling algorithm.
"""

import jax
import jax.numpy as jnp
from jax import random

def gibbs_sampler(
    key: jax.Array,
    initial_state: jax.Array,
    num_samples: int,
    log_prob_fn: callable,
    update_fns: list[callable]
) -> jax.Array:
    """
    This function runs the Gibbs sampler.

    Args:
        key: JAX PRNG key.
        initial_state: Initial state of the sampler.
        num_samples: Number of samples to generate.
        log_prob_fn: Log probability function of the target distribution.
        update_fns: List of functions, each updating one component of the state.

    Returns:
        Array of samples.
    """
    
    def body_fn(i, state_and_samples):
        current_state, samples = state_and_samples
        
        new_state = current_state
        for j, update_fn in enumerate(update_fns):
            key_comp, _ = random.split(random.fold_in(key, i * len(update_fns) + j))
            new_state = update_fn(key_comp, new_state, log_prob_fn)
        
        samples = samples.at[i].set(new_state)
        return new_state, samples

    samples = jnp.zeros((num_samples,) + initial_state.shape)
    _, final_samples = jax.lax.fori_loop(
        0, num_samples, body_fn, (initial_state, samples)
    )
    
    return final_samples

if __name__ == "__main__":
    # Example Usage: Sampling from a 2D Gaussian
    def gaussian_log_prob(state):
        # Simple 2D Gaussian with independent components
        return -0.5 * (jnp.sum(state**2))

    def update_component_0(key, state, log_prob_fn):
        # Update first component (e.g., using a Gaussian proposal)
        # For a simple Gibbs, assume conditional is known or approximated
        # Here, we'll just add some noise for demonstration
        new_val = state.at[0].set(state[0] + random.normal(key) * 0.1)
        return new_val

    def update_component_1(key, state, log_prob_fn):
        # Update second component
        new_val = state.at[1].set(state[1] + random.normal(key) * 0.1)
        return new_val

    key = random.PRNGKey(0)
    initial_state = jnp.array([0.5, -0.5])
    num_samples = 1000
    update_functions = [update_component_0, update_component_1]

    samples = gibbs_sampler(key, initial_state, num_samples, gaussian_log_prob, update_functions)
    print("Gibbs samples shape:", samples.shape)
    print("Mean of samples:", jnp.mean(samples, axis=0))
    print("Std of samples:", jnp.std(samples, axis=0))
