"""Test deserialization and metrics calculation from actual SMC run data."""

import sys
from pathlib import Path

import jax.numpy as jnp

from proteinsmc.io import read_lineage_data
from proteinsmc.utils import metrics


def test_deserialization_and_metrics(output_dir: str = "oed_results") -> None:
  """Test reading SMC output and calculating metrics from it."""
  output_path = Path(output_dir)

  # Find all arrayrecord files
  arrayrecord_files = sorted(output_path.glob("data_*.arrayrecord"))

  if not arrayrecord_files:
    print(f"❌ No arrayrecord files found in {output_dir}")
    print(
      "   Run a quick test first: python src/proteinsmc/oed/run.py --num_initial_experiments 1 --num_oed_iterations 0"
    )
    sys.exit(1)

  print(f"✅ Found {len(arrayrecord_files)} arrayrecord files")
  test_file = arrayrecord_files[0]
  print(f"\n📂 Testing file: {test_file.name}")
  print(f"   Size: {test_file.stat().st_size} bytes")

  # Read the lineage data
  try:
    records = list(read_lineage_data(str(test_file)))
    print(f"\n✅ Successfully deserialized {len(records)} records")
  except Exception as e:
    print(f"\n❌ Deserialization failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

  if not records:
    print("❌ No records found in file")
    sys.exit(1)

  # Examine the first and last records
  first_record = records[0]
  last_record = records[-1]

  print("\n📊 First record:")
  print(f"   Keys: {list(first_record.keys())}")
  print(f"   sequences shape: {first_record['sequences'].shape}")
  print(f"   fitness shape: {first_record['fitness'].shape}")
  print(f"   step: {first_record['step']}")

  print("\n📊 Last record:")
  print(f"   sequences shape: {last_record['sequences'].shape}")
  print(f"   fitness shape: {last_record['fitness'].shape}")
  print(f"   step: {last_record['step']}")

  # Extract sequences and fitness from first and last steps
  initial_sequences = first_record["sequences"]
  initial_fitness = first_record["fitness"]
  final_sequences = last_record["sequences"]
  final_fitness = last_record["fitness"]

  print("\n📈 Data shapes:")
  print(f"   Initial sequences: {initial_sequences.shape} (dtype={initial_sequences.dtype})")
  print(f"   Initial fitness: {initial_fitness.shape} (dtype={initial_fitness.dtype})")
  print(f"   Final sequences: {final_sequences.shape} (dtype={final_sequences.dtype})")
  print(f"   Final fitness: {final_fitness.shape} (dtype={final_fitness.dtype})")

  # Calculate metrics
  print("\n🧮 Calculating metrics...")

  # 1. Information gain (Jeffreys divergence between fitness distributions)
  # Note: We need to create histograms of fitness values, not use raw fitness
  # Or use a different metric. Let's try both approaches:

  # Approach 1: Histogram-based
  n_bins = 20
  initial_fitness_hist, _ = jnp.histogram(initial_fitness, bins=n_bins, density=False)
  final_fitness_hist, _ = jnp.histogram(final_fitness, bins=n_bins, density=False)
  initial_fitness_hist = initial_fitness_hist.astype(jnp.float32)
  final_fitness_hist = final_fitness_hist.astype(jnp.float32)

  info_gain_hist = metrics.jeffreys_divergence(initial_fitness_hist, final_fitness_hist)

  # Approach 2: Simple statistics
  initial_mean = jnp.mean(initial_fitness)
  initial_std = jnp.std(initial_fitness)
  final_mean = jnp.mean(final_fitness)
  final_std = jnp.std(final_fitness)

  print(f"\n   Initial fitness: mean={initial_mean:.4f}, std={initial_std:.4f}")
  print(f"   Final fitness: mean={final_mean:.4f}, std={final_std:.4f}")
  print(f"   Information gain (histogram-based): {info_gain_hist:.6f}")

  # 2. Barrier crossing frequency
  fitness_history = jnp.array([rec["fitness"] for rec in records])
  print(f"   Fitness history shape: {fitness_history.shape}")
  barrier_freq = metrics.calculate_barrier_crossing_frequency(fitness_history)
  print(f"   Barrier crossing frequency: {barrier_freq:.2f}")

  # 3. Final sequence entropy
  final_entropy = metrics.shannon_entropy(final_sequences)
  print(f"   Final sequence entropy: {final_entropy:.4f}")

  # 4. JSD from original population
  # Get the alphabet size from sequence values
  q = int(jnp.max(jnp.concatenate([initial_sequences.ravel(), final_sequences.ravel()]))) + 1
  print(f"   Detected alphabet size q={q}")

  p = jnp.bincount(initial_sequences.ravel(), minlength=q).astype(jnp.float32)
  q_dist = jnp.bincount(final_sequences.ravel(), minlength=q).astype(jnp.float32)
  jsd = metrics.jensen_shannon_divergence(p, q_dist)
  print(f"   JSD from original population: {jsd:.6f}")

  # Check for NaNs
  print("\n🔍 NaN check:")
  print(f"   Info gain is NaN: {jnp.isnan(info_gain_hist)}")
  print(f"   Barrier freq is NaN: {jnp.isnan(barrier_freq)}")
  print(f"   Final entropy is NaN: {jnp.isnan(final_entropy)}")
  print(f"   JSD is NaN: {jnp.isnan(jsd)}")

  if jnp.isnan(info_gain_hist):
    print("\n⚠️  Info gain is NaN - debugging:")
    print(f"   Initial hist: {initial_fitness_hist}")
    print(f"   Final hist: {final_fitness_hist}")
    print(f"   Sum initial: {jnp.sum(initial_fitness_hist)}")
    print(f"   Sum final: {jnp.sum(final_fitness_hist)}")

  if jnp.isnan(jsd):
    print("\n⚠️  JSD is NaN - debugging:")
    print(f"   p: {p}")
    print(f"   q_dist: {q_dist}")
    print(f"   Sum p: {jnp.sum(p)}")
    print(f"   Sum q: {jnp.sum(q_dist)}")

  print("\n✅ All metrics calculated successfully!")


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--output_dir",
    type=str,
    default="oed_results",
    help="Directory containing arrayrecord files",
  )
  args = parser.parse_args()

  test_deserialization_and_metrics(args.output_dir)
