#!/usr/bin/env python3
"""View and analyze OED tracking data.

This script loads the OED manifest and generates a summary of the experiments.
"""

import argparse
import json
from pathlib import Path

from proteinsmc.oed.tracking import create_oed_summary, load_oed_manifest


def main() -> None:
  """View OED tracking data."""
  parser = argparse.ArgumentParser(description="View OED tracking data")
  parser.add_argument(
    "output_dir",
    type=str,
    help="Directory containing OED results",
  )
  parser.add_argument(
    "--summary",
    action="store_true",
    help="Generate and display summary",
  )
  parser.add_argument(
    "--list",
    action="store_true",
    help="List all experiments",
  )
  args = parser.parse_args()

  output_path = Path(args.output_dir)

  if args.summary:
    print("=== OED Experiment Summary ===")
    summary = create_oed_summary(output_path)
    print(json.dumps(summary, indent=2))
    print(f"\nSummary saved to: {output_path / 'oed_summary.json'}")

  if args.list:
    print("\n=== OED Experiments ===")
    records = load_oed_manifest(output_path)
    for i, record in enumerate(records, 1):
      print(f"\n--- Experiment {i} ---")
      print(f"UUID: {record['run_uuid']}")
      print(f"Phase: {record['phase']}")
      print(f"Iteration: {record['iteration']}")
      print(f"Timestamp: {record['timestamp']}")
      print(f"Design: {record['design']}")
      print(f"Result: {record['result']}")

  if not args.summary and not args.list:
    # Default: show basic stats
    records = load_oed_manifest(output_path)
    print(f"Total experiments: {len(records)}")
    seeding = sum(1 for r in records if r["phase"] == "seeding")
    optimization = sum(1 for r in records if r["phase"] == "optimization")
    print(f"Seeding: {seeding}")
    print(f"Optimization: {optimization}")
    print("\nUse --summary or --list for more details")


if __name__ == "__main__":
  main()
