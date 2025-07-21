from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import uuid_utils as uuid
from safetensors.flax import serialize

if TYPE_CHECKING:
  from proteinsmc.models.sampler_base import BaseSamplerConfig


class DataWriter:
  """Handles buffering and batched writing of experiment results asynchronously."""

  def __init__(self, output_dir: Path, run_id: uuid.UUID, batch_size: int = 100) -> None:
    self.output_dir = output_dir
    self.run_id = run_id
    self.batch_size = batch_size
    self.results_buffer: list[Any] = []
    self.scalar_log_path = self.output_dir / f"{self.run_id}_metrics.jsonl"
    self._batch_index = 0
    # Use a thread pool to make file I/O non-blocking for the main process
    self.executor = ThreadPoolExecutor(max_workers=1)

  def log_scalars(self, step_metrics: dict) -> None:
    """Append a dictionary of scalar metrics to the JSONL file."""
    with open(self.scalar_log_path, "a") as f:
      f.write(json.dumps(step_metrics) + "\n")

  def step(self, step_result: Any) -> None:
    """Process a single step result, dispatching a write job if the buffer is full."""
    self.results_buffer.append(jax.device_get(step_result))
    if len(self.results_buffer) >= self.batch_size:
      self._dispatch_write()

  def _dispatch_write(self) -> None:
    """Submit the current buffer to the executor to be written to a file."""
    if not self.results_buffer:
      return

    # Move the data to a local variable so the buffer can be cleared immediately
    data_to_write = list(self.results_buffer)
    self.results_buffer.clear()

    # Submit the write operation to the background thread
    self.executor.submit(self._write_batch_thread, data_to_write, self._batch_index)
    self._batch_index += 1

  def _write_batch_thread(self, data: list[Any], batch_index: int) -> None:
    """The actual function that runs in the background thread."""
    batched_pytree = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *data)
    filename = self.output_dir / f"{self.run_id}_batch_{batch_index}.safetensors"
    serialize(batched_pytree, str(filename))

  def close(self) -> None:
    """Writes any remaining data and shuts down the executor."""
    self._dispatch_write()
    self.executor.shutdown(wait=True)  # Wait for all pending writes to complete


class RunManager:
  """A context manager to set up and tear down a single experiment run."""

  def __init__(
    self,
    output_dir: str | Path,
    config: BaseSamplerConfig,
    batch_size: int = 100,
  ) -> None:
    self.output_dir = Path(output_dir)
    self.config = config
    self.batch_size = batch_size
    self.run_id = uuid.uuid7()

  def __enter__(self) -> DataWriter:
    """Initializes the run: creates directory, saves metadata, returns a writer."""
    self.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = self.output_dir / f"{self.run_id}_metadata.json"
    with open(metadata_path, "w") as f:
      json.dump({"run_id": str(self.run_id), "config": asdict(self.config)}, f, indent=2)
    self.writer = DataWriter(self.output_dir, self.run_id, self.batch_size)
    return self.writer

  def __exit__(self, exc_type, exc_val, exc_tb):
    """Ensures all data is written at the end of the run."""
    self.writer.close()
