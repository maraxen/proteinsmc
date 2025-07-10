
import jax
import jax.numpy as jnp
import chex
import pytest

from proteinsmc.sampling.mcmc import make_random_mutation_proposal_fn, mcmc_sampler


@pytest.fixture
def simple_sequence():
  return jnp.array([0, 0, 0, 0], dtype=jnp.int8)


def test_make_random_mutation_proposal_fn(simple_sequence):
  key = jax.random.PRNGKey(0)
  n_states = 4
  proposal_fn = make_random_mutation_proposal_fn(n_states)

  proposed_seq = proposal_fn(key, simple_sequence)

  chex.assert_shape(proposed_seq, simple_sequence.shape)
  chex.assert_equal(jnp.sum(proposed_seq != simple_sequence), 1)


def test_mcmc_sampler_output_shape(simple_sequence):
  key = jax.random.PRNGKey(0)
  num_samples = 100

  def log_prob_fn(key, x):
    return -jnp.sum(x).astype(jnp.float32)

  proposal_fn = make_random_mutation_proposal_fn(n_states=4)

  output = mcmc_sampler(key, simple_sequence, num_samples, log_prob_fn, proposal_fn)

  chex.assert_shape(output.samples, (num_samples, *simple_sequence.shape))


def test_mcmc_sampler_converges():
  """Test that the MCMC sampler converges to a simple target distribution."""
  key = jax.random.PRNGKey(42)
  sequence_length = 5
  n_states = 2

  def log_prob_fn(key, seq):
    return jnp.sum(seq) * 10.0

  proposal_fn = make_random_mutation_proposal_fn(n_states=n_states)

  initial_state = jnp.zeros(sequence_length, dtype=jnp.int8)
  num_samples = 1000

  output = mcmc_sampler(key, initial_state, num_samples, log_prob_fn, proposal_fn)

  burn_in = num_samples // 2
  mean_state_value = jnp.mean(output.samples[burn_in:])

  chex.assert_trees_all_close(mean_state_value, 1.0, atol=0.1)
