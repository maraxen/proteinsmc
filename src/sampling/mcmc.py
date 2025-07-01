"""
This module implements the Metropolis-Hastings MCMC sampling algorithm.
"""

import jax
import jax.numpy as jnp
from jax import random


def mcmc_sampler(
    key: jax.Array,
    initial_state: jax.Array,
    num_samples: int,
    log_prob_fn: callable,
    proposal_fn: callable,
) -> jax.Array:
    """
    This function runs the Metropolis-Hastings MCMC sampler.

    Args:
        key: JAX PRNG key.
        initial_state: Initial state of the sampler.
        num_samples: Number of samples to generate.
        log_prob_fn: Log probability function of the target distribution.
        proposal_fn: Proposal function to generate new states.

    Returns:
        Array of samples.
    """

    def body_fn(i, state_and_samples):
        current_state, samples = state_and_samples
        
        key_proposal, key_accept = random.split(random.fold_in(key, i))
        
        proposed_state = proposal_fn(key_proposal, current_state)
        
        current_log_prob = log_prob_fn(current_state)
        proposed_log_prob = log_prob_fn(proposed_state)
        
        # Metropolis-Hastings acceptance ratio
        acceptance_ratio = jnp.exp(proposed_log_prob - current_log_prob)
        
        # Accept or reject the proposed state
        accept = random.uniform(key_accept) < acceptance_ratio
        
        next_state = jnp.where(accept, proposed_state, current_state)
        samples = samples.at[i].set(next_state)
        
        return next_state, samples

    samples = jnp.zeros((num_samples,) + initial_state.shape)
    _, final_samples = jax.lax.fori_loop(
        0, num_samples, body_fn, (initial_state, samples)
    )
    
    return final_samples

if __name__ == "__main__":
    # Example Usage: Sampling from a 1D Gaussian
    def gaussian_log_prob(x):
        return -0.5 * jnp.sum(x**2)

    def proposal_fn(key, state):
        return state + random.normal(key) * 0.5

    key = random.PRNGKey(0)
    initial_state = jnp.array([0.0])
    num_samples = 10000

    samples = mcmc_sampler(
        key, initial_state, num_samples, gaussian_log_prob, proposal_fn
    )
    print("MCMC samples shape:", samples.shape)
    print("Mean of samples:", jnp.mean(samples, axis=0))
    print("Std of samples:", jnp.std(samples, axis=0))
