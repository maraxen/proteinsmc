"""Unit tests for BaseSamplerConfig and SamplerOutputProtocol in sampler_base.py.

These tests cover type validation, value validation, and property behavior.
Mocks are used for FitnessEvaluator, MemoryConfig, and AnnealingConfig dependencies.

Run with:
  pytest tests/models/test_sampler_base.py
"""


import pytest
from proteinsmc.models.sampler_base import BaseSamplerConfig, SamplerOutputProtocol

from proteinsmc.models.fitness import FitnessEvaluator
from proteinsmc.models.memory import MemoryConfig
from proteinsmc.models.annealing import AnnealingConfig


# Mocks for dependencies
class DummyFitnessEvaluator(FitnessEvaluator):
  pass

class DummyMemoryConfig(MemoryConfig):
  pass

class DummyAnnealingConfig(AnnealingConfig):
  pass

@pytest.fixture
def valid_config_kwargs(monkeypatch):
  """Fixture providing valid arguments for BaseSamplerConfig.
  
  Args:
    monkeypatch: pytest monkeypatch fixture for patching methods.
    
  Returns:
    dict: Valid arguments for BaseSamplerConfig instantiation.
    
  Example:
    >>> config = BaseSamplerConfig(**valid_config_kwargs)
  """
  # Monkeypatch __post_init__ methods to avoid validation/initialization
  monkeypatch.setattr(FitnessEvaluator, "__post_init__", lambda self: None)
  
  return dict(
    prng_seed=123,
    sampler_type="smc",
    seed_sequence="ACDEFGHIKLMNPQRSTVWY",
    num_samples=10,
    n_states=20,
    mutation_rate=0.05,
    diversification_ratio=0.2,
    sequence_type="protein",
    fitness_evaluator=DummyFitnessEvaluator(
      fitness_functions=()),
    memory_config=DummyMemoryConfig(),
    annealing_config=DummyAnnealingConfig(
      annealing_fn="dummy_fn",
      beta_max=1.0,
      n_steps=100,
    ),
  )

def test_basesamplerconfig_valid_instantiation(valid_config_kwargs):
  """Test that BaseSamplerConfig can be instantiated with valid arguments.

  Args:
    valid_config_kwargs (dict): Valid arguments for BaseSamplerConfig.

  Returns:
    None

  Raises:
    AssertionError: If instantiation fails or fields are not set correctly.

  Example:
    >>> test_basesamplerconfig_valid_instantiation(valid_config_kwargs)
  """
  config = BaseSamplerConfig(**valid_config_kwargs)
  assert config.prng_seed == 123
  assert config.sampler_type == "smc"
  assert config.seed_sequence == "ACDEFGHIKLMNPQRSTVWY"
  assert config.num_samples == 10
  assert config.n_states == 20
  assert config.mutation_rate == 0.05
  assert config.diversification_ratio == 0.2
  assert config.sequence_type == "protein"
  assert isinstance(config.fitness_evaluator, DummyFitnessEvaluator)
  assert isinstance(config.memory_config, DummyMemoryConfig)
  assert isinstance(config.annealing_config, DummyAnnealingConfig)

def test_basesamplerconfig_generations_property(valid_config_kwargs):
  """Test that generations property returns num_samples.

  Args:
    valid_config_kwargs (dict): Valid arguments for BaseSamplerConfig.

  Returns:
    None

  Raises:
    AssertionError: If generations does not match num_samples.

  Example:
    >>> test_basesamplerconfig_generations_property(valid_config_kwargs)
  """
  config = BaseSamplerConfig(**valid_config_kwargs)
  assert config.generations == config.num_samples

@pytest.mark.parametrize(
  "field,value,error_type,error_msg",
  [
  ("n_states", 0, ValueError, "n_states must be positive."),
  ("num_samples", 0, ValueError, "generations must be positive."),
  ("mutation_rate", -0.1, ValueError, r"mutation_rate must be in \[0\.0, 1\.0\]\."),
  ("mutation_rate", 1.1, ValueError, r"mutation_rate must be in \[0\.0, 1\.0\]\."),
  ("diversification_ratio", -0.1, ValueError, r"diversification_ratio must be in \[0\.0, 1\.0\]\."),
  ("diversification_ratio", 1.1, ValueError, r"diversification_ratio must be in \[0\.0, 1\.0\]\."),
  ("sequence_type", "invalid", ValueError, "sequence_type must be 'protein' or 'nucleotide'."),
  ]
)
def test_basesamplerconfig_value_validation(field, value, error_type, error_msg, valid_config_kwargs):
  """Test that invalid values raise ValueError with correct message.

  Args:
    field (str): Field name to set to an invalid value.
    value: Invalid value for the field.
    error_type (Exception): Expected exception type.
    error_msg (str): Expected error message.
    valid_config_kwargs (dict): Valid arguments for BaseSamplerConfig.

  Returns:
    None

  Raises:
    AssertionError: If exception is not raised or message does not match.

  Example:
    >>> test_basesamplerconfig_value_validation("n_states", 0, ValueError, "n_states must be positive.", valid_config_kwargs)
  """
  kwargs = valid_config_kwargs.copy()
  kwargs[field] = value
  with pytest.raises(error_type, match=error_msg):
    BaseSamplerConfig(**kwargs)

def test_basesamplerconfig_validate_types_type_errors(valid_config_kwargs):
  """Test that _validate_types raises TypeError for wrong types.

  Args:
    valid_config_kwargs (dict): Valid arguments for BaseSamplerConfig.

  Returns:
    None

  Raises:
    AssertionError: If TypeError is not raised for wrong types.

  Example:
    >>> test_basesamplerconfig_validate_types_type_errors(valid_config_kwargs)
  """
  # seed_sequence not str
  kwargs = valid_config_kwargs.copy()
  kwargs["seed_sequence"] = 123
  config = BaseSamplerConfig(**valid_config_kwargs)
  with pytest.raises(TypeError, match="seed_sequence must be a string."):
    BaseSamplerConfig(**kwargs)
    

  # n_states not int
  kwargs = valid_config_kwargs.copy()
  kwargs["n_states"] = "20"
  with pytest.raises(TypeError, match="n_states must be an integer."):
    BaseSamplerConfig(**kwargs)

  # num_samples not int
  kwargs = valid_config_kwargs.copy()
  kwargs["num_samples"] = "10"
  with pytest.raises(TypeError, match="generations must be an integer."):
    BaseSamplerConfig(**kwargs)

  # mutation_rate not float
  kwargs = valid_config_kwargs.copy()
  kwargs["mutation_rate"] = "0.1"
  with pytest.raises(TypeError, match="mutation_rate must be a float."):
    BaseSamplerConfig(**kwargs)

  # diversification_ratio not float
  kwargs = valid_config_kwargs.copy()
  kwargs["diversification_ratio"] = "0.0"
  with pytest.raises(TypeError, match="diversification_ratio must be a float."):
    BaseSamplerConfig(**kwargs)

  # fitness_evaluator not FitnessEvaluator
  kwargs = valid_config_kwargs.copy()
  kwargs["fitness_evaluator"] = object()
  with pytest.raises(TypeError, match="fitness_evaluator must be a FitnessEvaluator instance."):
    BaseSamplerConfig(**kwargs)

  # memory_config not MemoryConfig
  kwargs = valid_config_kwargs.copy()
  kwargs["memory_config"] = object()
  with pytest.raises(TypeError, match="memory_config must be a MemoryConfig instance."):
    BaseSamplerConfig(**kwargs)

def test_basesamplerconfig_additional_config_fields(valid_config_kwargs):
  """Test that additional_config_fields returns an empty dict by default.

  Args:
    valid_config_kwargs (dict): Valid arguments for BaseSamplerConfig.

  Returns:
    None

  Raises:
    AssertionError: If additional_config_fields is not an empty dict.

  Example:
    >>> test_basesamplerconfig_additional_config_fields(valid_config_kwargs)
  """
  config = BaseSamplerConfig(**valid_config_kwargs)
  assert config.additional_config_fields == {}

def test_sampleroutputprotocol_properties():
  """Test the default properties of SamplerOutputProtocol.

  Args:
    None

  Returns:
    None

  Raises:
    AssertionError: If default property values do not match expected.

  Example:
    >>> test_sampleroutputprotocol_properties()
  """
  class DummyOutput(SamplerOutputProtocol):
    pass

  output = DummyOutput()
  assert output.per_gen_stats_metrics == {}
  assert output.summary_stats_metrics == {}
  assert output.output_type_name == "SamplerOutput"