"""Test script to validate deserialization of OED experiment outputs."""

import logging
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from proteinsmc.io import read_lineage_data
from proteinsmc.models.sampler_base import SamplerOutput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Find the most recent output file
output_dir = Path("oed_results")
arrayrecord_files = sorted(output_dir.glob("data_*.arrayrecord"), key=lambda p: p.stat().st_mtime)

if not arrayrecord_files:
    logger.error("No arrayrecord files found in %s", output_dir)
    sys.exit(1)

# Test the most recent file
test_file = arrayrecord_files[-1]
logger.info("Testing deserialization of: %s", test_file)
logger.info("File size: %d bytes", test_file.stat().st_size)

try:
    # Create a skeleton for deserialization
    skeleton = SamplerOutput(
        step=jnp.array(0, dtype=jnp.int32),
        sequences=jnp.array([], dtype=jnp.int8),
        fitness=jnp.array([], dtype=jnp.float32),
        key=jax.random.PRNGKey(0),
    )
    records = list(read_lineage_data(str(test_file), skeleton))
    logger.info("Successfully read %d records", len(records))

    if records:
        first_record = records[0]
        logger.info("\nFirst record structure:")
        logger.info("  Type: %s", type(first_record))

        if hasattr(first_record, "__dataclass_fields__"):
            import dataclasses
            fields = (
                dataclasses.fields(first_record)
                if dataclasses.is_dataclass(first_record)
                else []
            )
            keys_info = [f.name for f in fields]
            logger.info("  Keys (fields): %s", keys_info)

            for field in fields:
                value = getattr(first_record, field.name)
                logger.info("  %s: type=%s, shape=%s",
                           field.name,
                           type(value).__name__,
                           getattr(value, "shape", "N/A"))
        else:
            logger.info("  Not a dataclass")

        # Check a few more records
        logger.info("\nChecking all %d records...", len(records))
        for i, record in enumerate(records):
            if not isinstance(record, SamplerOutput):
                logger.error("Record %d is not a SamplerOutput: %s", i, type(record))
            else:
                seq_shape = getattr(record.sequences, "shape", None)
                fitness_shape = getattr(record.fitness, "shape", None)
                logger.info("Record %d: sequences.shape=%s, fitness.shape=%s",
                           i, seq_shape, fitness_shape)

        logger.info("\n✅ Deserialization test PASSED")

except Exception:
    logger.exception("❌ Deserialization test FAILED")
    sys.exit(1)
