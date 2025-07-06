import jax
import jax.numpy as jnp
import pytest

from proteinsmc.sampling.gibbs import (
  gibbs_sampler,
  make_gibbs_update_fns,
)
from proteinsmc.utils.fitness import FitnessEvaluator, FitnessFunction, make_sequence_log_prob_fn


@pytest.fixture
def fitness_evaluator():
  """Creates a mock fitness evaluator that returns a single score per sequence."""

  def mock_fitness_func(key, seq, const_val=5.0):
    return const_val

  fitness_func = FitnessFunction(
    func=mock_fitness_func, input_type="protein", args={"const_val": 5.0}, name="mock"
  )
  return FitnessEvaluator(fitness_functions=[fitness_func])


def test_make_sequence_log_prob_fn(fitness_evaluator):
  """Test that the created log_prob function returns the expected fitness."""
  log_prob_fn = make_sequence_log_prob_fn(fitness_evaluator, sequence_type="protein")

  test_seq = jnp.ones((2, 10))
  log_prob = log_prob_fn(test_seq)
  print(log_prob)
  assert log_prob.shape == (2,)
  assert jnp.all(log_prob == 5.0)

  test_batch = jnp.ones((5, 10))
  log_probs = log_prob_fn(test_batch)
  assert log_probs.shape == (5,)
  assert jnp.all(log_probs == 5.0)


def test_make_gibbs_update_fns():
  """Test the creation of Gibbs update functions."""
  update_fns = make_gibbs_update_fns(sequence_length=5, n_states=4)
  assert len(update_fns) == 5

  update_pos_0 = update_fns[0]
  key = jax.random.PRNGKey(0)
  seq = jnp.zeros((5,), dtype=jnp.int8)

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

  def log_prob_fn(seq):
    return jnp.where(seq[0] == seq[1], 1.0, -100.0)

  update_fns = make_gibbs_update_fns(sequence_length=sequence_length, n_states=n_states)

  initial_state = jnp.array([0, 1])
  num_samples = 100

  samples = gibbs_sampler(key, initial_state, num_samples, log_prob_fn, update_fns)

  assert samples.shape == (num_samples, sequence_length)

  matches = jnp.sum(samples[-50:, 0] == samples[-50:, 1])
  assert matches > 40
