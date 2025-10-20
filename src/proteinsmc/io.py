"""I/O utilities for simulation tracking using ArrayRecord and msgpack serialization."""

import uuid
from collections.abc import Callable
from typing import Any

import msgpack
import msgpack_numpy
import numpy as np
from array_record.python.array_record_module import ArrayRecordReader, ArrayRecordWriter
from etils import epath
from jaxtyping import Array, UInt8

msgpack_numpy.patch()
UUID_BYTE_LENGTH = 36


def create_writer_callback(path: str) -> tuple[ArrayRecordWriter, Callable]:
  """Create an ArrayRecordWriter and a msgpack/lineage callback.

  Returns:
    A tuple containing:
      - The ArrayRecordWriter instance (must be closed manually).
      - The callback function for use with io_callback.

  """
  writer = ArrayRecordWriter(epath.Path(path))

  def writer_callback(
    payload: dict[str, Array],
    parent_uuid_bytes: Array,
    entry_type: Array,
  ) -> UInt8:
    """Instantiate callback function (closure).

    This function is executed by io_callback.
    """
    new_uuid = str(uuid.uuid4())
    parent_uuid_str = parent_uuid_bytes.tobytes().decode("utf-8").strip("\x00")
    if not parent_uuid_str:
      parent_uuid_str = None

    full_record = {
      "uuid": new_uuid,
      "parent_uuid": parent_uuid_str,
      "payload": payload,
      "entry_type": entry_type,
    }

    packed_bytes = msgpack.packb(full_record)

    writer.write({"data": packed_bytes})

    return np.array(new_uuid.encode("utf-8"), dtype=np.uint8)

  return writer, writer_callback


def read_lineage_data(path: str) -> dict[str, Any]:
  """Read and deserialize all records from a lineage file.

  Args:
      path: The path to the ArrayRecord file.

  Returns:
      A dictionary mapping UUIDs to their corresponding records.

  """
  reader = ArrayRecordReader(epath.Path(path))
  records_list = list(reader.read())

  history = {}
  for record in records_list:
    packed_bytes = record["data"]
    full_record = msgpack.unpackb(packed_bytes)
    history[full_record["uuid"]] = full_record

  return history
