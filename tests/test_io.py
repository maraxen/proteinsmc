import json
import subprocess
import uuid
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from proteinsmc.io import create_writer_callback
from proteinsmc.models.fitness import FitnessEvaluator, FitnessFunction
from proteinsmc.models.smc import SMCConfig


@patch("proteinsmc.models.sampler_base.BaseSamplerConfig._initialize_device_mesh", return_value=None)
def test_create_writer_callback_creates_metadata_file(mock_mesh, tmp_path: Path):
    """Test that create_writer_callback creates a metadata.json file with the correct content."""
    run_uid = uuid.uuid4().hex

    # Create a dummy FitnessFunction
    dummy_fitness_function = FitnessFunction(name="dummy", n_states=4, kwargs={})

    # Create a real FitnessEvaluator instance
    fitness_evaluator = FitnessEvaluator(fitness_functions=[dummy_fitness_function])

    # Create a real SMCConfig instance
    config = SMCConfig(
        seed_sequence="ATGC",
        fitness_evaluator=fitness_evaluator,
    )

    with patch("subprocess.check_output", return_value=b"test_hash"):
        writer, _ = create_writer_callback(
            path=str(tmp_path), run_uid=run_uid, config=config
        )

    metadata_path = tmp_path / "metadata.json"
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text())

    assert metadata["run_uid"] == run_uid
    assert metadata["sampler_type"] == "smc"
    assert "start_timestamp" in metadata
    assert "config" in metadata
    assert metadata["config"]["seed_sequence"] == "ATGC"
    assert metadata["git_commit_hash"] == "test_hash"
    assert metadata["config"]["mesh"] is None

    writer.close()
