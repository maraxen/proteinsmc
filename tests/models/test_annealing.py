"""Unit tests for the annealing schedule data structures.

Tests cover the AnnealingKwargs TypedDict and AnnealingConfig dataclass.
"""

from typing import Any

import pytest
import dataclasses
from proteinsmc.models.annealing import AnnealingConfig, AnnealingFuncSignature, AnnealingKwargs


def test_annealing_kwargs_typeddict():
  """Test that AnnealingKwargs TypedDict accepts correct keys and types.

  Args:
    None

  Returns:
    None

  Raises:
    AssertionError: If the TypedDict does not accept the correct keys and types.

  Example:
    >>> test_annealing_kwargs_typeddict()

  """
  kwargs: AnnealingKwargs = {"current_step": 5, "_context": {"foo": "bar"}}
  assert kwargs["current_step"] == 5
  assert kwargs["_context"] == {"foo": "bar"}

  # _context can be None
  kwargs2: AnnealingKwargs = {"current_step": 0, "_context": None}
  assert kwargs2["_context"] is None


def test_annealing_func_signature_type():
  """Test that AnnealingFuncSignature accepts a function with the correct signature.

  Args:
    None

  Returns:
    None

  Raises:
    AssertionError: If the function signature does not match.

  Example:
    >>> test_annealing_func_signature_type()

  """

  def dummy_schedule(current_step: int, _context: Any | None) -> float:
    return float(current_step)

  # Should be assignable to AnnealingFuncSignature
  func: AnnealingFuncSignature = dummy_schedule # type: ignore[assignment]
  result = func(current_step=3, _context=None) # type: ignore[arg-type]
  assert isinstance(result, float)
  assert result == 3.0


def test_annealing_config_creation():
  """Test creation and attribute access of AnnealingConfig dataclass.

  Args:
    None

  Returns:
    None

  Raises:
    AssertionError: If attributes are not set or accessed correctly.

  Example:
    >>> test_annealing_config_creation()

  """
  config = AnnealingConfig(
    annealing_fn="linear",
    beta_max=10.0,
    n_steps=100,
    schedule_args={"foo": 1, "bar": 2},
  )
  assert config.annealing_fn == "linear"
  assert config.beta_max == 10.0
  assert config.n_steps == 100
  assert config.schedule_args == {"foo": 1, "bar": 2}


def test_annealing_config_default_schedule_args():
  """Test that AnnealingConfig.schedule_args defaults to an empty dict.

  Args:
    None

  Returns:
    None

  Raises:
    AssertionError: If the default value is not an empty dict.

  Example:
    >>> test_annealing_config_default_schedule_args()

  """
  config = AnnealingConfig(
    annealing_fn="exp",
    beta_max=5.0,
    n_steps=50,
  )
  assert config.schedule_args == {}


def test_annealing_config_immutable():
  """Test that AnnealingConfig is immutable (frozen=True).

  Args:
    None

  Returns:
    None

  Raises:
    AssertionError: If the dataclass is not frozen.

  Example:
    >>> test_annealing_config_immutable()

  """
  config = AnnealingConfig(
    annealing_fn="exp",
    beta_max=5.0,
    n_steps=50,
  )
  with pytest.raises(dataclasses.FrozenInstanceError):
    config.beta_max = 42.0 # type: ignore[assignment]
