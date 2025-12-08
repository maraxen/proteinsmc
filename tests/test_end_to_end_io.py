"""End-to-end I/O tests."""

import json
from pathlib import Path
from unittest.mock import patch

import jax.numpy as jnp
import pytest
from proteinsmc.models import SMCConfig
from proteinsmc.models.fitness import FitnessEvaluator, FitnessFunction
from proteinsmc.models.memory import MemoryConfig
from proteinsmc.runner import run_experiment
from proteinsmc.io import read_lineage_data
from proteinsmc.utils.serialization import create_sampler_output_skeleton

@pytest.fixture
def mock_fitness_function():
    def dummy_fitness(key, seq, context):
        # seq: (length, alpha)
        # Returns stacked fitness (1,)
        return jnp.zeros((1,), dtype=jnp.float32)
    return dummy_fitness

def test_smc_run_io(tmp_path: Path, mock_fitness_function):
    """Run a short SMC experiment and verify output."""

    with patch("proteinsmc.runner.get_fitness_function", return_value=mock_fitness_function):
        config = SMCConfig(
            sampler_type="smc",
            seed_sequence="AAAAAA", # length 6
            num_samples=2, # 2 steps
            population_size=5,
            n_states=4, # nucleotide
            fitness_evaluator=FitnessEvaluator(
                 fitness_functions=(
                     FitnessFunction(name="dummy_fn", n_states=4),
                 ),
            ),
            memory_config=MemoryConfig(),
            algorithm="BaseSMC",
            resampling_approach="systematic",
            sequence_type="nucleotide",
        )

        output_dir = tmp_path / "output"
        run_experiment(config, output_dir, seed=42)

    # Verify output
    files = list(output_dir.glob("data.arrayrecord"))
    assert len(files) == 1
    data_file = files[0]

    # Read metadata
    with open(output_dir / "metadata.json") as f:
        meta = json.load(f)
        assert meta["sampler_type"] == "smc"

    # Create skeleton
    skeleton = create_sampler_output_skeleton(config)

    # Verify skeleton shape
    # Sequences: (pop, len) -> (5, 6)
    assert skeleton.sequences.shape == (5, 6)

    # Read data
    records = list(read_lineage_data(str(data_file), skeleton))

    assert len(records) == 2

    # Check step 1
    assert records[0].step == 1
    assert records[0].sequences.shape == (5, 6)

    # Check step 2
    assert records[1].step == 2
