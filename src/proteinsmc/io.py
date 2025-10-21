"""I/O utilities for simulation tracking using ArrayRecord and msgpack serialization."""

import datetime
import json
import subprocess
import uuid
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import jax.numpy as jnp
import msgpack
import msgpack_numpy
import numpy as np
from array_record.python.array_record_module import ArrayRecordReader, ArrayRecordWriter
from etils import epath
from jaxtyping import Array, PyTree

from proteinsmc.models.sampler_base import BaseSamplerConfig

msgpack_numpy.patch()


def _json_serializer(obj: Any) -> Any:
  """Custom serializer for objects that are not directly JSON serializable."""
  if isinstance(obj, (np.ndarray, jnp.ndarray)):
    return obj.tolist()
  if isinstance(obj, (np.integer, jnp.integer)):
    return int(obj)
  if isinstance(obj, (np.floating, jnp.floating)):
    return float(obj)
  if isinstance(obj, bytes):
    return obj.decode("utf-8")
  raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def create_writer_callback(
  path: str,
  run_uid: str,
  config: BaseSamplerConfig,
) -> tuple[ArrayRecordWriter, Callable]:
  """Create an ArrayRecordWriter, a metadata file, and a msgpack/lineage callback."""
  output_dir = epath.Path(path)
  output_dir.mkdir(parents=True, exist_ok=True)

  try:
    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("ascii")
  except (subprocess.CalledProcessError, FileNotFoundError):
    git_hash = None

  metadata = {
    "run_uid": run_uid,
    "sampler_type": config.sampler_type,
    "start_timestamp": datetime.datetime.now().isoformat(),
    "config": asdict(config),
    "git_commit_hash": git_hash,
  }
  metadata_path = output_dir / "metadata.json"
  with metadata_path.open("w") as f:
    json.dump(metadata, f, indent=2, default=_json_serializer)

  writer_path = output_dir / "records.ar"
  writer = ArrayRecordWriter(str(writer_path))

  def writer_callback(
    pytree_payload: PyTree[Array],
  ) -> None:
    """Instantiate callback function (closure).

    This function is executed by io_callback.
    """
    packed_bytes = msgpack.packb(pytree_payload)
    writer.write({"data": packed_bytes, "run_uid": run_uid})

  return writer, writer_callback


def read_lineage_data(path: str) -> list[dict[str, Any]]:
  """Read and deserialize all records from a lineage file.

  Args:
      path: The path to the ArrayRecord file.

  Returns:
      A list of records.

  """
  reader = ArrayRecordReader(epath.Path(path))
  records_list = list(reader.read())

  history = []
  for record in records_list:
    packed_bytes = record["data"]
    full_record = msgpack.unpackb(packed_bytes)
    history.append(full_record)

  return history
