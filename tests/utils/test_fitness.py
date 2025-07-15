
import jax
import jax.numpy as jnp
import chex
import pytest
from proteinsmc.utils.fitness import chunked_calculate_population_fitness

from proteinsmc.utils import (
  calculate_population_fitness,
  FitnessEvaluator,
  FitnessFunction,
)

def make_mock_fitness_function(mock_func, **kwargs):
  """Create a mock fitness function for testing."""
  
  def mock_fitness_func(key, sequences):
    """Mock fitness function that returns a constant value."""
    return mock_func(key, sequences, **kwargs)

  return mock_fitness_func


def mock_protein_fitness_func(key, sequences, arg1):
  """Mock fitness function that operates on protein sequences."""
  return jnp.full((sequences.shape[0],), arg1).mean(axis=-1)


def mock_nucleotide_fitness_func(key, sequences, arg1):
  """Mock fitness function that operates on nucleotide sequences."""
  return jnp.full((sequences.shape[0],), arg1).mean(axis=-1)

made_mock_protein_fitness_func = make_mock_fitness_function(
  mock_protein_fitness_func, arg1=2.0
)
made_mock_nucleotide_fitness_func = make_mock_fitness_function(
  mock_nucleotide_fitness_func, arg1=3.0
)

@pytest.fixture
def protein_ff() -> FitnessFunction:
  """Fixture to create a mock protein fitness function."""
  return FitnessFunction(
    func=made_mock_protein_fitness_func,
    input_type="protein",
    name="mock_protein",
  )

@pytest.fixture
def nucleotide_ff() -> FitnessFunction:
  """Fixture to create a mock nucleotide fitness function."""
  return FitnessFunction(
    func=made_mock_nucleotide_fitness_func,
    input_type="nucleotide",
    name="mock_nucleotide",
  )


def test_fitness_function_init(protein_ff):
  """Test FitnessFunction initialization."""
  chex.assert_equal(protein_ff.name, "mock_protein")


def test_fitness_function_invalid_func():
  """Test that FitnessFunction raises ValueError for a non-callable func."""
  with pytest.raises(TypeError):
    FitnessFunction(
      func="not_a_function",  # type: ignore
      input_type="protein",
      name="invalid",
    )


def test_fitness_function_invalid_input_type():
  """Test that FitnessFunction raises ValueError for an invalid input_type."""
  with pytest.raises(ValueError):
    FitnessFunction(
      func=made_mock_protein_fitness_func,
      input_type="invalid_type",  # type: ignore
      name="invalid",
    )


def test_fitness_evaluator_init(protein_ff):
  """Test FitnessEvaluator initialization."""
  fe = FitnessEvaluator(fitness_functions=(protein_ff,))
  chex.assert_equal(len(fe.fitness_functions), 1)


def test_fitness_evaluator_no_functions():
  """Test that FitnessEvaluator raises ValueError if no functions are provided."""
  with pytest.raises(ValueError):
    FitnessEvaluator(fitness_functions=())


def test_fitness_evaluator_get_functions_by_type(protein_ff, nucleotide_ff):
  """Test retrieving functions by type from FitnessEvaluator."""
  fe = FitnessEvaluator(fitness_functions=(protein_ff, nucleotide_ff,))
  protein_funcs = fe.get_functions_by_type("protein")
  nucleotide_funcs = fe.get_functions_by_type("nucleotide")
  chex.assert_equal(len(protein_funcs), 1)
  chex.assert_equal(protein_funcs[0].name, "mock_protein")
  chex.assert_equal(len(nucleotide_funcs), 1)
  chex.assert_equal(nucleotide_funcs[0].name, "mock_nucleotide")


def test_calculate_population_fitness_nucleotide(protein_ff, nucleotide_ff):
  """Test fitness calculation for a population of nucleotide sequences."""
  key = jax.random.PRNGKey(0)
  population = jnp.array([[0, 1, 2, 3, 0, 1]])

  evaluator = FitnessEvaluator(fitness_functions=(protein_ff, nucleotide_ff,))

  combined_fitness, components = calculate_population_fitness(
    key, population, "nucleotide", evaluator
  )

  chex.assert_trees_all_close(components[0], jnp.array([2.0]))
  chex.assert_trees_all_close(components[1], jnp.array([3.0]))
  chex.assert_trees_all_close(combined_fitness, jnp.array([5.0]))
  
def test_chunked_calculate_population_fitness(protein_ff, nucleotide_ff):
  """Test chunked_calculate_population_fitness for memory-efficient batch fitness."""

  key = jax.random.PRNGKey(42)
  population = jnp.tile(jnp.array([[0, 1, 2, 3, 0, 1]]), (10, 1))

  evaluator = FitnessEvaluator(fitness_functions=(protein_ff, nucleotide_ff,))

  chunk_size = 4

  combined_fitness, components = chunked_calculate_population_fitness(
    key, population, evaluator, "nucleotide", chunk_size=chunk_size
  )

  assert combined_fitness.shape == (10,)
  assert components.shape == (2, 10)

  chex.assert_trees_all_close(components[0], jnp.full((10,), 2.0))
  chex.assert_trees_all_close(components[1], jnp.full((10,), 3.0))
  chex.assert_trees_all_close(combined_fitness, jnp.full((10,), 5.0))
