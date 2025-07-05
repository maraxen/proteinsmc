import jax
import jax.numpy as jnp
import pytest
from jax import random

from proteinsmc.sampling.smc import (
  SMCConfig,
  smc,
)
from proteinsmc.utils.annealing_schedules import linear_schedule
from proteinsmc.utils.fitness import FitnessEvaluator, FitnessFunction


# Mock fitness function for testing
def mock_fitness_fn(key, sequences, sequence_type):
  # Simple fitness: sum of integer values in the sequence
  return jnp.sum(sequences, axis=-1), {
    "cai": jnp.mean(sequences, axis=-1),
    "mpnn": -jnp.mean(sequences, axis=-1),
  }


@pytest.fixture
def setup_smc():
  key = random.PRNGKey(0)
  initial_sequence = "MKY"
  config = SMCConfig(
    sequence_length=len(initial_sequence),
    population_size=10,
    mutation_rate=0.1,
    sequence_type="protein",
    evolve_as="nucleotide",
    fitness_evaluator=FitnessEvaluator(
      fitness_functions=[
        FitnessFunction(func=mock_fitness_fn, input_type="nucleotide", name="mock", args={})
      ]
    ),
  )
  annealing_schedule_fn = linear_schedule
  annealing_schedule_args = (1.0, 5)
  generations = 5
  diversification_ratio = 0.2

  return (
    key,
    config,
    initial_sequence,
    annealing_schedule_fn,
    annealing_schedule_args,
    generations,
    diversification_ratio,
  )


def test_smc_output_shapes(setup_smc):
  (
    key,
    config,
    initial_sequence,
    annealing_schedule_fn,
    annealing_schedule_args,
    generations,
    diversification_ratio,
  ) = setup_smc

  key_smc, key_init = random.split(key)

  output = smc(
    prng_key_smc_steps=key_smc,
    initial_population_key=key_init,
    diversification_ratio=diversification_ratio,
    initial_sequence=initial_sequence,
    sequence_type=config.sequence_type,
    evolve_as=config.evolve_as,
    mutation_rate=config.mutation_rate,
    annealing_schedule_function=annealing_schedule_fn,
    annealing_schedule_args=annealing_schedule_args,
    population_size=config.population_size,
    generations=generations,
    fitness_evaluator=config.fitness_evaluator,
  )

  assert output.mean_combined_fitness_per_gen.shape == (generations,)
  assert output.max_combined_fitness_per_gen.shape == (generations,)
  assert output.mean_cai_per_gen.shape == (generations,)
  assert output.mean_mpnn_score_per_gen.shape == (generations,)
  assert output.entropy_per_gen.shape == (generations,)
  assert output.beta_per_gen.shape == (generations,)
  assert output.ess_per_gen.shape == (generations,)
  assert isinstance(output.final_logZhat, (float, jax.Array))


@pytest.mark.parametrize(
  "sequence_type, evolve_as",
  [("protein", "nucleotide"), ("nucleotide", "nucleotide")],
)
def test_smc_sequence_types(setup_smc, sequence_type, evolve_as):
  (
    key,
    config,
    initial_sequence,
    annealing_schedule_fn,
    annealing_schedule_args,
    generations,
    diversification_ratio,
  ) = setup_smc

  if sequence_type == "nucleotide":
    initial_sequence = "AUGAAAUAC"

  config = config._replace(sequence_type=sequence_type, evolve_as=evolve_as)

  key_smc, key_init = random.split(key)

  output = smc(
    prng_key_smc_steps=key_smc,
    initial_population_key=key_init,
    diversification_ratio=diversification_ratio,
    initial_sequence=initial_sequence,
    sequence_type=config.sequence_type,
    evolve_as=config.evolve_as,
    mutation_rate=config.mutation_rate,
    annealing_schedule_function=annealing_schedule_fn,
    annealing_schedule_args=annealing_schedule_args,
    population_size=config.population_size,
    generations=generations,
    fitness_evaluator=config.fitness_evaluator,
  )
  assert output is not None
