import jax
import jax.numpy as jnp
import pytest

from proteinsmc.utils.fitness import (
  FitnessEvaluator,
  FitnessFunction,
  calculate_population_fitness,
)


# Mock fitness functions
def mock_protein_fitness_func(key, sequences, arg1):
  """Mock fitness function that operates on protein sequences."""
  return jnp.full(sequences.shape[0], arg1)


def mock_nucleotide_fitness_func(key, sequences, arg2):
  """Mock fitness function that operates on nucleotide sequences."""
  return jnp.full(sequences.shape[0], arg2)


@pytest.fixture
def protein_ff():
  """Fixture for a protein-based fitness function."""
  return FitnessFunction(
    func=mock_protein_fitness_func,
    input_type="protein",
    args={"arg1": 2.0},
    name="mock_protein",
  )


@pytest.fixture
def nucleotide_ff():
  """Fixture for a nucleotide-based fitness function."""
  return FitnessFunction(
    func=mock_nucleotide_fitness_func,
    input_type="nucleotide",
    args={"arg2": 3.0},
    name="mock_nucleotide",
  )


def test_fitness_function_init(protein_ff):
  """Test FitnessFunction initialization."""
  assert protein_ff.name == "mock_protein"
  assert protein_ff.is_active


def test_fitness_function_invalid_func():
  """Test that FitnessFunction raises ValueError for a non-callable func."""
  with pytest.raises(ValueError):
    FitnessFunction(
      func="not_a_function",  # type: ignore
      input_type="protein",
      args={},
      name="invalid",
    )


def test_fitness_function_invalid_input_type():
  """Test that FitnessFunction raises ValueError for an invalid input_type."""
  with pytest.raises(ValueError):
    FitnessFunction(
      func=mock_protein_fitness_func,
      input_type="invalid_type",  # type: ignore
      args={},
      name="invalid",
    )


def test_fitness_evaluator_init(protein_ff):
  """Test FitnessEvaluator initialization."""
  fe = FitnessEvaluator(fitness_functions=[protein_ff])
  assert len(fe.fitness_functions) == 1


def test_fitness_evaluator_no_functions():
  """Test that FitnessEvaluator raises ValueError if no functions are provided."""
  with pytest.raises(ValueError):
    FitnessEvaluator(fitness_functions=[])


def test_fitness_evaluator_get_active_functions(protein_ff, nucleotide_ff):
  """Test retrieving active functions from FitnessEvaluator."""
  nucleotide_ff.is_active = False
  fe = FitnessEvaluator(fitness_functions=[protein_ff, nucleotide_ff])
  active_funcs = fe.get_active_functions()
  assert len(active_funcs) == 1
  assert active_funcs[0].name == "mock_protein"


def test_fitness_evaluator_get_functions_by_type(protein_ff, nucleotide_ff):
  """Test retrieving functions by type from FitnessEvaluator."""
  fe = FitnessEvaluator(fitness_functions=[protein_ff, nucleotide_ff])
  protein_funcs = fe.get_functions_by_type("protein")
  nucleotide_funcs = fe.get_functions_by_type("nucleotide")
  assert len(protein_funcs) == 1
  assert protein_funcs[0].name == "mock_protein"
  assert len(nucleotide_funcs) == 1
  assert nucleotide_funcs[0].name == "mock_nucleotide"


def test_calculate_population_fitness_nucleotide(protein_ff, nucleotide_ff):
  """Test fitness calculation for a population of nucleotide sequences."""
  key = jax.random.PRNGKey(0)
  population = jnp.array([[0, 1, 2, 3, 0, 1]])  # ACGTAC

  evaluator = FitnessEvaluator(fitness_functions=[protein_ff, nucleotide_ff])

  combined_fitness, components = calculate_population_fitness(
    key, population, "nucleotide", evaluator
  )

  assert "mock_nucleotide" in components
  assert "mock_protein" in components
  assert jnp.allclose(components["mock_nucleotide"], jnp.array([3.0]))
  assert jnp.allclose(components["mock_protein"], jnp.array([2.0]))
  # Default combination is weighted sum with weights of 1.0
  assert jnp.allclose(combined_fitness, jnp.array([5.0]))
