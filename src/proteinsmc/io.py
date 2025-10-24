"""I/O utilities for simulation tracking using ArrayRecord and msgpack serialization."""

import json
import shutil
import subprocess
from collections.abc import Callable, Generator
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import msgpack_numpy
from array_record.python.array_record_module import ArrayRecordReader, ArrayRecordWriter
from flax.serialization import msgpack_restore, msgpack_serialize, to_state_dict
from jaxtyping import PyTree

from proteinsmc.models.sampler_base import SamplerOutput

msgpack_numpy.patch()


def get_git_commit_hash() -> str | None:
  """Get the current git commit hash if available.

  Returns:
      Git commit hash as a string, or None if not in a git repository.

  """
  git_executable = shutil.which("git")
  if git_executable is None:
    return None

  try:
    # S603: subprocess call - check for execution of untrusted input
    # This is safe because we're using a hardcoded git command with validated path
    result = subprocess.run(  # noqa: S603
      [git_executable, "rev-parse", "HEAD"],
      capture_output=True,
      text=True,
      check=True,
      timeout=5,
    )
    return result.stdout.strip()
  except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
    return None


def create_metadata_file(config: object, output_path: Path) -> None:
  """Create a metadata JSON file with run configuration and git info.

  Args:
      config: The sampler configuration object.
      output_path: Path where the metadata file should be saved.

  """
  # Extract config data - handle both dataclass and regular objects
  config_data: dict[str, Any] | str
  if is_dataclass(config) and not isinstance(config, type):
    # Manually extract config fields to avoid pickling JAX Mesh/Device objects
    # We can't use asdict() because it deepcopies all values, including unpicklable ones
    config_data = {}
    for field_info in fields(config):
      if field_info.name == "mesh":
        # Skip the mesh field - it contains JAX Device objects that can't be pickled
        continue
      value = getattr(config, field_info.name)
      # For nested dataclasses, convert to dict or string representation
      if is_dataclass(value) and not isinstance(value, type):
        try:
          config_data[field_info.name] = asdict(value)  # type: ignore[arg-type]
        except (TypeError, RecursionError):
          config_data[field_info.name] = str(value)
      else:
        config_data[field_info.name] = value
  else:
    config_data = str(config)

  metadata = {
    "timestamp": datetime.now().isoformat(),  # noqa: DTZ005
    "git_commit_hash": get_git_commit_hash(),
    "sampler_type": getattr(config, "sampler_type", "unknown"),
    "config": config_data,
  }

  metadata_path = output_path / "metadata.json"
  with metadata_path.open("w") as f:
    json.dump(metadata, f, indent=2, default=str)


def create_writer_callback(path: str) -> tuple[ArrayRecordWriter, Callable]:
  """Create an ArrayRecordWriter and a msgpack callback.

  Returns:
    A tuple containing:
      - The ArrayRecordWriter instance (must be closed manually).
      - The callback function for use with io_callback.

  """
  writer = ArrayRecordWriter(path)

  def writer_callback(
    sampler_output: SamplerOutput,
  ) -> None:
    """Write a payload to the ArrayRecord file.

    This function is executed by io_callback.

    Args:
        sampler_output: The SamplerOutput data to write (will be serialized with msgpack).

    """
    state_dict = to_state_dict(sampler_output)
    serialized = msgpack_serialize(state_dict)
    writer.write(serialized)

  return writer, writer_callback


def read_lineage_data(path: str) -> Generator[PyTree, None, None]:
  """Read and deserialize all records from a lineage file.

  Args:
      path: The path to the ArrayRecord file.

  Returns:
      A generator yielding deserialized records.

  """
  reader = ArrayRecordReader(path)

  try:
    num_records = reader.num_records()
    for _ in range(num_records):
      packed_bytes = reader.read()  # read() with no args returns next record
      full_record = msgpack_restore(packed_bytes)
      yield full_record
  except IndexError:
    # No more records or empty file
    pass
