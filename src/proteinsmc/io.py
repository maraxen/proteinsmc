"""I/O utilities for simulation tracking using ArrayRecord and msgpack serialization."""

import json
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import msgpack
import msgpack_numpy
from array_record.python.array_record_module import ArrayRecordReader, ArrayRecordWriter
from jaxtyping import PyTree

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
    config_data = asdict(config)  # type: ignore[arg-type]
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
    pytree_payload: PyTree,
  ) -> None:
    """Write a payload to the ArrayRecord file.

    This function is executed by io_callback.

    Args:
        pytree_payload: The payload data to write (will be serialized with msgpack).

    """
    packed_bytes = msgpack.packb(pytree_payload)
    writer.write({"data": packed_bytes})

  return writer, writer_callback


def read_lineage_data(path: str) -> dict[str, Any]:
  """Read and deserialize all records from a lineage file.

  Args:
      path: The path to the ArrayRecord file.

  Returns:
      A list of deserialized records.

  """
  reader = ArrayRecordReader(path)
  records_list = list(reader.read())

  history = []
  for record in records_list:
    packed_bytes = record["data"]
    full_record = msgpack.unpackb(packed_bytes)
    history.append(full_record)

  return {"records": history}
