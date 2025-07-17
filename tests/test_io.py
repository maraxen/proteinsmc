"""Tests for the I/O module."""

import json
import tempfile
import time
from pathlib import Path
import uuid

import jax.numpy as jnp
from proteinsmc.io import RunManager
from proteinsmc.models.base import BaseSamplerConfig
from proteinsmc.models.fitness import FitnessEvaluator, FitnessFunction

def test_run_manager_and_data_writer():
    """
    Tests that the RunManager and DataWriter create the correct files
    and that the non-blocking writes complete.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Create a dummy config
        dummy_config = BaseSamplerConfig(
            sampler_type="test_sampler",
            seed_sequence="M",
            generations=10,
            n_states=20,
            mutation_rate=0.1,
            diversification_ratio=0.5,
            sequence_type="protein",
            fitness_evaluator=FitnessEvaluator(
                fitness_functions=(
                    FitnessFunction(name="test_func", input_type="protein"),
                )
            ),
            memory_config=None # type: ignore
        )

        run_id = None
        with RunManager(output_dir, dummy_config, batch_size=5) as writer:
            run_id = writer.run_id
            assert isinstance(run_id, uuid.UUID)

            # Log some scalars
            writer.log_scalars({"step": 0, "metric": 1.0})
            writer.log_scalars({"step": 1, "metric": 2.0})

            # Log some step results (PyTrees)
            for i in range(12):
                writer.step({"data": jnp.ones((2, 3)) * i})
        
        # The RunManager __exit__ calls writer.close(), which blocks until
        # all futures are complete.

        # Verify that all files were created
        metadata_file = output_dir / f"{run_id}_metadata.json"
        metrics_file = output_dir / f"{run_id}_metrics.jsonl"
        batch_0_file = output_dir / f"{run_id}_batch_0.safetensors"
        batch_1_file = output_dir / f"{run_id}_batch_1.safetensors"
        batch_2_file = output_dir / f"{run_id}_batch_2.safetensors" # 2 leftover items

        assert metadata_file.exists()
        assert metrics_file.exists()
        assert batch_0_file.exists()
        assert batch_1_file.exists()
        assert batch_2_file.exists()

        # Verify content of metadata
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            assert metadata["run_id"] == str(run_id)
            assert metadata["config"]["sampler_type"] == "test_sampler"

        # Verify content of metrics
        with open(metrics_file, "r") as f:
            lines = f.readlines()
            assert len(lines) == 2
            assert json.loads(lines[0])["metric"] == 1.0
            assert json.loads(lines[1])["metric"] == 2.0
