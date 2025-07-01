"""
This module implements the Hamiltonian Monte Carlo (HMC) sampling algorithm.
"""

import jax
import jax.numpy as jnp
from jax import random


def hmc_sampler(
    key: jax.Array,
    initial_position: jax.Array,
    num_samples: int,
    log_prob_fn: callable,
    step_size: float,
    num_leapfrog_steps: int,
) -> jax.Array:
    """
    This function runs the Hamiltonian Monte Carlo (HMC) sampler.

    Args:
        key: JAX PRNG key.
        initial_position: Initial position of the sampler.
        num_samples: Number of samples to generate.
        log_prob_fn: Log probability function of the target distribution.
        step_size: Integration step size for leapfrog.
        num_leapfrog_steps: Number of leapfrog steps to take.

    Returns:
        Array of samples.
    """

    def leapfrog(q, p, log_prob_fn, step_size, num_steps):
        grad_log_prob = jax.grad(log_prob_fn)

        def body_fn(i, carry):
            q, p = carry
            # Half step for momentum
            p_half = p + step_size * grad_log_prob(q) / 2.0
            # Full step for position
            q_new = q + step_size * p_half
            # Half step for momentum
            p_new = p_half + step_size * grad_log_prob(q_new) / 2.0
            return q_new, p_new

        final_q, final_p = jax.lax.fori_loop(0, num_steps, body_fn, (q, p))
        return final_q, final_p

    def hmc_step(carry, _):
        current_q, current_log_prob, key = carry

        key_momentum, key_leapfrog, key_accept = random.split(key, 3)

        # Resample momentum
        p0 = random.normal(key_momentum, shape=current_q.shape)

        # Perform leapfrog steps
        q_new, p_new = leapfrog(
            current_q, p0, log_prob_fn, step_size, num_leapfrog_steps
        )

        # Calculate Hamiltonian
        current_hamiltonian = -current_log_prob + 0.5 * jnp.sum(p0**2)
        proposed_log_prob = log_prob_fn(q_new)
        proposed_hamiltonian = -proposed_log_prob + 0.5 * jnp.sum(p_new**2)

        # Metropolis-Hastings acceptance
        acceptance_ratio = jnp.exp(current_hamiltonian - proposed_hamiltonian)
        accept = random.uniform(key_accept) < acceptance_ratio

        next_q = jnp.where(accept, q_new, current_q)
        next_log_prob = jnp.where(accept, proposed_log_prob, current_log_prob)

        return (next_q, next_log_prob, key_accept), next_q

    # Initialize
    initial_log_prob = log_prob_fn(initial_position)

    # Run sampler
    _, samples = jax.lax.scan(
        hmc_step,
        (initial_position, initial_log_prob, key),
        None,
        length=num_samples
    )

    return samples

if __name__ == "__main__":
    # Example Usage: Sampling from a 2D Gaussian
    def gaussian_log_prob(x):
        return -0.5 * jnp.sum(x**2)

    key = random.PRNGKey(0)
    initial_position = jnp.array([0.0, 0.0])
    num_samples = 1000
    step_size = 0.1
    num_leapfrog_steps = 10

    samples = hmc_sampler(
        key, initial_position, num_samples, gaussian_log_prob, 
        step_size, num_leapfrog_steps
    )
    print("HMC samples shape:", samples.shape)
    print("Mean of samples:", jnp.mean(samples, axis=0))
    print("Std of samples:", jnp.std(samples, axis=0))
