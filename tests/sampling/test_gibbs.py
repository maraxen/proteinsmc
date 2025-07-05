import jax
import jax.numpy as jnp
import pytest

from proteinsmc.sampling.gibbs import (
  gibbs_sampler,
  make_gibbs_update_fns,
  make_sequence_log_prob_fn,
)
from proteinsmc.utils.fitness import FitnessEvaluator, FitnessFunction


# Mock fitness function for testing
def mock_fitness_func(key, sequences, const_val):
  return jnp.full(sequences.shape[0], const_val)


@pytest.fixture
def fitness_evaluator():
  """Provides a mock FitnessEvaluator."""
  ff = FitnessFunction(
    func=mock_fitness_func, input_type="protein", args={"const_val": 5.0}, name="mock"
  )
  return FitnessEvaluator(fitness_functions=[ff])


def test_make_sequence_log_prob_fn(fitness_evaluator):
  """Test that the created log_prob function returns the expected fitness."""
  log_prob_fn = make_sequence_log_prob_fn(fitness_evaluator, evolve_as="protein")
  test_seq = jnp.ones((10,))
  log_prob = log_prob_fn(test_seq)
  assert log_prob == 5.0

  # Test batching
  test_batch = jnp.ones((5, 10))
  log_probs = log_prob_fn(test_batch)
  assert log_probs.shape == (5,)
  assert jnp.all(log_probs == 5.0)


def test_make_gibbs_update_fns():
  """Test the creation of Gibbs update functions."""
  update_fns = make_gibbs_update_fns(sequence_length=5, n_states=4, evolve_as="protein")
  assert len(update_fns) == 5

  # Test a single update function
  update_pos_0 = update_fns[0]
  key = jax.random.PRNGKey(0)
  seq = jnp.zeros((5,), dtype=jnp.int32)

  # Mock log_prob_fn that strongly prefers state 3 at position 0
  def deterministic_log_prob(s):
    return jnp.where(s[0] == 3, 100.0, 0.0)

  updated_seq = update_pos_0(key, seq, deterministic_log_prob)
  assert updated_seq.shape == seq.shape
  assert updated_seq[0] == 3
  assert jnp.array_equal(updated_seq[1:], seq[1:])


def test_gibbs_sampler():
  """Test the full Gibbs sampler on a simple correlated distribution."""
  key = jax.random.PRNGKey(42)
  sequence_length = 2
  n_states = 4

  # Target distribution: log_prob is high if seq[0] == seq[1], else low
  def log_prob_fn(seq):
    return jnp.where(seq[0] == seq[1], 1.0, -100.0)

  update_fns = make_gibbs_update_fns(
    sequence_length=sequence_length, n_states=n_states, evolve_as="protein"
  )

  initial_state = jnp.array([0, 1])  # Start in a low-probability state
  num_samples = 100

  samples = gibbs_sampler(key, initial_state, num_samples, log_prob_fn, update_fns)

  assert samples.shape == (num_samples, sequence_length)

  # After burn-in, most samples should have seq[0] == seq[1]
  # Check the last 50 samples
  matches = jnp.sum(samples[-50:, 0] == samples[-50:, 1])
  assert matches > 40  # High probability of matching
