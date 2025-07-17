"""Utilities for saving batches of JAX/Flax experiment outputs to single files and processing them.

This module implements a two-stage process:
1.  A batch of individual experiment runs is "stacked" into a single PyTree
    and serialized to a single raw bytes file using flax.serialization.
    This ensures perfect, lossless storage of the complete experiment state
    for multiple runs in one file.
2.  A batch file can be processed to extract key analytical metrics, which
    are then compiled into an efficient, columnar Parquet file for analysis
    and visualization across all experiments in the batch.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import (
  Any,
  Callable,
  TypeVar,
  get_args,
  get_origin,
)

import dill
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from flax.serialization import (
  from_state_dict,
  msgpack_restore,
  register_serialization_state,
  to_bytes,
)

from proteinsmc.models import (
  AnnealingConfig,
  FitnessEvaluator,
  FitnessFunction,
  MemoryConfig,
  ParallelReplicaSMCOutput,
  SMCCarryState,
  SMCConfig,
  SMCOutput,
)

logger = logging.getLogger(__name__)


T = TypeVar("T", SMCOutput, ParallelReplicaSMCOutput)


def _dill_callable_to_bytes(func: Callable) -> bytes:
  """Serialize a callable to bytes using dill."""
  try:
    return dill.dumps(func)
  except Exception:
    logger.exception("Failed to dill callable %r", func)
    raise


def _dill_bytes_to_callable(data: bytes) -> Callable:
  """Deserialize bytes back to a callable using dill."""
  try:
    return dill.loads(data)  # noqa: S301
  except Exception:
    logger.exception("Failed to load callable from bytes")
    raise


def _convert_dataclass_to_serializable_dict(obj: Any) -> Any:
  """Recursively convert dataclass instances into serializable dictionaries.

  Handles callable attributes by dill-serializing them to bytes.
  """
  output = None
  if isinstance(obj, (jax.Array, np.ndarray)):
    output = obj
  elif callable(obj) and not isinstance(obj, (type,)):
    output = _dill_callable_to_bytes(obj)

  if output is None:
    if is_dataclass(obj) and not isinstance(obj, type):
      d = asdict(obj)
      processed_dict = {}
      for k, v in d.items():
        processed_dict[k] = _convert_dataclass_to_serializable_dict(v)
      output = processed_dict

    elif isinstance(obj, (list, tuple)):
      output = type(obj)(_convert_dataclass_to_serializable_dict(item) for item in obj)

    elif isinstance(obj, dict):
      output = {k: _convert_dataclass_to_serializable_dict(v) for k, v in obj.items()}

  return output


def _from_serializable_dict_recursive(cls: type, state: dict[str, Any]) -> Any:  # noqa: C901
  """Recursively convert a serializable dictionary state back to a dataclass instance.

  Handles dill-deserializing bytes back to callables.
  """

  def _handle_callable(field_type: Any, field_value: Any) -> Any:
    if (
      get_origin(field_type) is Callable
      or getattr(field_type, "__origin__", None) is Callable
      or (hasattr(field_type, "__args__") and Callable in get_args(field_type))
    ) and isinstance(field_value, bytes):
      return _dill_bytes_to_callable(field_value)
    return None

  def _handle_dataclass(field_type: Any, field_value: Any) -> Any:
    if is_dataclass(field_type) and isinstance(field_value, dict):
      return _from_serializable_dict_recursive(type(field_type), field_value)
    return None

  def _handle_sequence(field_type: Any, field_value: Any) -> Any:
    if get_origin(field_type) in (list, tuple) and isinstance(field_value, (list, tuple)):
      inner_type_args = get_args(field_type)
      if inner_type_args:
        if is_dataclass(inner_type_args[0]) and field_value:
          return type(field_value)(
            _from_serializable_dict_recursive(type(inner_type_args[0]), item)
            for item in field_value
          )
        if (
          get_origin(inner_type_args[0]) is Callable or Callable in get_args(inner_type_args[0])
        ) and all(isinstance(item, bytes) for item in field_value):
          return type(field_value)(_dill_bytes_to_callable(item) for item in field_value)
        return field_value
      return field_value
    return None

  if not is_dataclass(cls):
    if isinstance(state, bytes):
      try:
        return _dill_bytes_to_callable(state)
      except Exception:  # noqa: BLE001
        return state
    return state

  processed_state = {}
  for field_info in fields(cls):
    field_name = field_info.name
    field_value = state.get(field_name)
    field_type = field_info.type

    # Try each handler in order, assign if handled
    value = _handle_callable(field_type, field_value)
    if value is not None:
      processed_state[field_name] = value
      continue

    value = _handle_dataclass(field_type, field_value)
    if value is not None:
      processed_state[field_name] = value
      continue

    value = _handle_sequence(field_type, field_value)
    if value is not None:
      processed_state[field_name] = value
      continue

    processed_state[field_name] = field_value

  # Special handling for SMCOutput's input_config, as it will be a list of dicts if stacked
  if cls is SMCOutput and "input_config" in state:
    config_value = state["input_config"]
    if isinstance(config_value, list):
      processed_state["input_config"] = [
        _from_serializable_dict_recursive(SMCConfig, cfg_dict) for cfg_dict in config_value
      ]
    elif isinstance(config_value, dict):
      processed_state["input_config"] = _from_serializable_dict_recursive(
        SMCConfig,
        config_value,
      )

  try:
    return cls(**processed_state)
  except TypeError:
    logger.exception(
      "Failed to unflatten %s with state %r",
      cls.__name__,
      processed_state,
    )
    raise


def _register_flax_serialization_for_all_types() -> None:
  """Register custom serialization for all relevant custom types."""
  all_dataclasses = [
    SMCConfig,
    MemoryConfig,
    AnnealingConfig,
    FitnessEvaluator,
    FitnessFunction,
    SMCCarryState,
    SMCOutput,
    ParallelReplicaSMCOutput,
  ]

  for cls_to_reg in all_dataclasses:
    register_serialization_state(
      cls_to_reg,
      ty_to_state_dict=_convert_dataclass_to_serializable_dict,
      ty_from_state_dict=lambda target, state: _from_serializable_dict_recursive(
        type(target),
        state,
      ),
    )


_register_flax_serialization_for_all_types()


def stack_outputs(outputs: list[T]) -> T:
  """Stack a list of PyTree objects into a single PyTree with a leading batch dimension.

  Args:
      outputs: A list of PyTree objects (e.g., SMCOutput instances) to stack.

  Returns:
      A single PyTree of the same type, where JAX arrays are stacked and other
      types (like strings) are collected into lists.

  Raises:
      ValueError: If the input list of outputs is empty.

  """
  if not outputs:
    msg = "Cannot stack an empty list of outputs."
    raise ValueError(msg)

  def stack_leaves(*leaves: Any) -> Any:
    """Stack JAX arrays or collect other types into a list."""
    if isinstance(leaves[0], (jax.Array,)) or hasattr(leaves[0], "shape"):
      return jnp.stack(leaves)
    return leaves[0] if len(leaves) == 1 else list(leaves)

  return jax.tree_util.tree_map(stack_leaves, *outputs)


def save_batch_output(
  outputs: list[T],
  batch_file_path: str | Path,
) -> None:
  """Stack and save a batch of experiment outputs to a single file.

  Args:
      outputs: A list of experiment output objects to save.
      batch_file_path: The path for the single output file (e.g., 'batch_01.flax').

  """
  batch_file_path = Path(batch_file_path)
  batch_file_path.parent.mkdir(parents=True, exist_ok=True)

  stacked_output = stack_outputs(outputs)
  serialized_batch = to_bytes(stacked_output)
  batch_file_path.write_bytes(serialized_batch)


def _get_target_type(type_name: str) -> type[T]:
  """Map a type name string to an actual class type.

  Args:
      type_name (str): The name of the type as a string.

  Returns:
      type[T]: The corresponding class type.

  Raises:
      TypeError: If the output_type name is unknown.

  """
  type_map = {
    "SMCOutput": SMCOutput,
    "ParallelReplicaSMCOutput": ParallelReplicaSMCOutput,
  }
  if type_name not in type_map:
    msg = f"Unknown output_type name: {type_name}"
    raise TypeError(msg)
  return type_map[type_name]


def load_batch_output(batch_file_path: str | Path) -> T:
  """Load and deserialize a batch of experiment outputs from a single file.

  This function intelligently inspects the file to determine the object type
  before deserializing, removing the need for a pre-made target object.

  Args:
      batch_file_path: The path to the batch file.

  Returns:
      T: The deserialized, stacked PyTree object.

  Raises:
      ValueError: If the output type cannot be determined or state_dict is incompatible.

  """
  batch_file_path = Path(batch_file_path)
  serialized_data = batch_file_path.read_bytes()

  state_dict = msgpack_restore(serialized_data)

  if "output_type" not in state_dict or not state_dict["output_type"]:
    msg = "Cannot determine output type from batch file."
    raise ValueError(msg)
  output_type_name = state_dict["output_type"][0]
  target_type = _get_target_type(output_type_name)

  dummy_target = target_type.__new__(target_type)
  if not isinstance(state_dict, dict):
    msg = "Incompatible state_dict structure."
    raise TypeError(msg)
  return from_state_dict(target=dummy_target, state=state_dict)  # type: ignore[return-value,call-arg]


def _convert_to_native_types(obj: Any) -> Any:
  """Recursively convert JAX/Numpy types in a PyTree to native Python types.

  Args:
      obj (Any): The object to convert.

  Returns:
      Any: The object with JAX/Numpy types converted to native Python types.

  """
  if isinstance(obj, (jax.Array, np.ndarray)):
    return obj.tolist()
  if isinstance(obj, np.generic):
    return obj.item()
  if isinstance(obj, (list, tuple)):
    return type(obj)(_convert_to_native_types(item) for item in obj)
  if isinstance(obj, dict):
    return {key: _convert_to_native_types(value) for key, value in obj.items()}
  if is_dataclass(obj) and not isinstance(obj, type):
    return _convert_to_native_types(asdict(obj))
  return obj


def _extract_analytical_data_from_stacked(stacked_output: T) -> list[dict[str, Any]]:
  """Extract key metrics from a stacked experiment batch for analysis.

  This function processes the stacked PyTree directly for efficiency.

  Args:
      stacked_output: The stacked experiment output object (SMCOutput or ParallelReplicaSMCOutput).

  Returns:
      list[dict[str, Any]]: A list of dictionaries, where each dictionary
                            represents a row in the final Parquet file.

  """
  all_records = []

  if isinstance(stacked_output, SMCOutput):
    output_data: SMCOutput = stacked_output
    batch_size = output_data.mean_combined_fitness_per_gen.shape[0]
    num_generations = output_data.beta_per_gen.shape[
      1
    ]  # Assuming beta_per_gen is (batch_size, num_gens)

    for i in range(batch_size):
      current_config = output_data.input_config[i]  # type: ignore[index]

      for j in range(num_generations):
        record = {
          "experiment_name": current_config.seed_sequence,
          "generation": j,
          "mean_fitness": output_data.mean_combined_fitness_per_gen[i, j].item(),
          "max_fitness": output_data.max_combined_fitness_per_gen[i, j].item(),
          "ess": output_data.ess_per_gen[i, j].item(),
          "beta": output_data.beta_per_gen[i, j].item(),
        }
        all_records.append(record)

  return all_records


def process_batch_to_parquet(
  batch_file_path: str | Path,
  parquet_path: str | Path,
) -> None:
  """Process a raw batch file into a single analytical Parquet file.

  Args:
      batch_file_path: The path to the input `.flax` batch file.
      parquet_path: The path for the output Parquet file.

  """
  try:
    stacked_output = load_batch_output(batch_file_path)
    records = _extract_analytical_data_from_stacked(stacked_output)

    if not records:
      return

    df = pl.DataFrame(records)
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(parquet_path, compression="snappy")

  except Exception:
    logger.exception("Failed to process batch file to Parquet")
    raise
