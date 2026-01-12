"""Test script to validate deserialization of OED experiment outputs."""

import logging
import sys
from pathlib import Path

from proteinsmc.io import read_lineage_data

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
  records = list(read_lineage_data(str(test_file)))
  logger.info("Successfully read %d records", len(records))

  if records:
    first_record = records[0]
    logger.info("\nFirst record structure:")
    logger.info("  Type: %s", type(first_record))
    keys_info = list(first_record.keys()) if isinstance(first_record, dict) else "Not a dict"
    logger.info("  Keys: %s", keys_info)

    if isinstance(first_record, dict):
      for key, value in first_record.items():
        logger.info(
          "  %s: type=%s, shape=%s", key, type(value).__name__, getattr(value, "shape", "N/A")
        )

    # Check a few more records
    logger.info("\nChecking all %d records...", len(records))
    for i, record in enumerate(records):
      if not isinstance(record, dict):
        logger.error("Record %d is not a dict: %s", i, type(record))
      elif "sequences" in record:
        seq_shape = record["sequences"].shape if hasattr(record["sequences"], "shape") else None
        fitness_shape = (
          record["fitness"].shape
          if "fitness" in record and hasattr(record["fitness"], "shape")
          else None
        )
        logger.info("Record %d: sequences.shape=%s, fitness.shape=%s", i, seq_shape, fitness_shape)

    logger.info("\n✅ Deserialization test PASSED")

except Exception:
  logger.exception("❌ Deserialization test FAILED")
  sys.exit(1)
