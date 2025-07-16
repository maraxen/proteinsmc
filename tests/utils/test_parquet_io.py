"""Tests for the experiment serialization and processing module."""

import datetime
from dataclasses import replace
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import polars as pl
import pytest


from proteinsmc.utils import parquet_io
from proteinsmc.utils.data_structures import (
    SMCConfig,
    SMCOutput,
    MemoryConfig,
    AnnealingScheduleConfig,
    FitnessEvaluator,
    FitnessFunction,
)


BASE_SMC_CONFIG = SMCConfig(
    seed_sequence="base", # This will be replaced per-run
    population_size=128,
    n_states=20,
    generations=10,
    mutation_rate=0.1,
    diversification_ratio=0.5,
    sequence_type="protein",
    annealing_schedule=AnnealingScheduleConfig(
        schedule_fn=lambda _, _a, x: x, beta_max=1.0, n_steps=10
    ),
    fitness_evaluator=FitnessEvaluator(
        fitness_functions=(
            FitnessFunction(
                func=lambda _key, seq: jnp.sum(seq, axis=-1, dtype=jnp.float32),
                input_type="protein",
                name="mock_fitness_function",
            ),
        )
    ),
    memory_config=MemoryConfig(population_chunk_size=32),
)

def _create_mock_smc_output(name: str, num_gens: int = 10) -> SMCOutput:
  """Factory for creating a mock SMCOutput instance with a unique config name."""
  key = jax.random.PRNGKey(hash(name))
  # Each run gets a config with a different name but is otherwise identical
  run_config = replace(BASE_SMC_CONFIG, seed_sequence=name)
  return SMCOutput(
    input_config=run_config,
    mean_combined_fitness_per_gen=jax.random.normal(key, (num_gens,)),
    max_combined_fitness_per_gen=jax.random.normal(key, (num_gens,)),
    entropy_per_gen=jax.random.uniform(key, (num_gens,)),
    beta_per_gen=jnp.linspace(0, 1, num_gens),
    ess_per_gen=jax.random.uniform(key, (num_gens,)),
    fitness_components_per_gen=jax.random.normal(key, (num_gens, 2)),
    final_logZhat=jnp.array(1.23, dtype=jnp.float32),
    final_amino_acid_entropy=jnp.array(4.56, dtype=jnp.float32),
  )


# --- Pytest Fixtures ---
@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
  """Creates a temporary directory for test artifacts."""
  return tmp_path


@pytest.fixture
def mock_smc_batch() -> list[SMCOutput]:
  """Creates a batch of mock SMCOutput objects."""
  return [_create_mock_smc_output(f"run_{i}") for i in range(3)]


@pytest.fixture(autouse=True)
def correct_pytree_definitions(monkeypatch):
    """
    Monkey-patches dataclasses to have correct PyTree definitions for testing.
    This is the core fix for the serialization and stacking issues.
    """
    # For SMCOutput, the config should be static (aux_data) and arrays dynamic (children)
    def smc_output_flatten(self):
        children = (
            self.input_config,
            self.mean_combined_fitness_per_gen,
            self.max_combined_fitness_per_gen,
            self.entropy_per_gen,
            self.beta_per_gen,
            self.ess_per_gen,
            self.fitness_components_per_gen,
            self.final_logZhat,
            self.final_amino_acid_entropy,
        )
        aux_data = {}
        return children, aux_data

    def smc_output_unflatten(cls, aux_data, children):
        # This unflatten method now correctly matches the flatten method.
        return cls(
            input_config=children[0],
            mean_combined_fitness_per_gen=children[1],
            max_combined_fitness_per_gen=children[2],
            entropy_per_gen=children[3],
            beta_per_gen=children[4],
            ess_per_gen=children[5],
            fitness_components_per_gen=children[6],
            final_logZhat=children[7],
            final_amino_acid_entropy=children[8],
            **aux_data,
        )

    monkeypatch.setattr(SMCOutput, "tree_flatten", smc_output_flatten)
    monkeypatch.setattr(SMCOutput, "tree_unflatten", classmethod(smc_output_unflatten))


    monkeypatch.setattr(SMCConfig, "tree_flatten", lambda self: ((), dict(self.__dict__)))
    monkeypatch.setattr(SMCConfig, "tree_unflatten", classmethod(lambda cls, aux, _: cls(**aux)))


# --- Test Functions ---

def test_stack_outputs_basic(mock_smc_batch: list[SMCOutput]):
  """Test the basic functionality of stacking outputs with a correct PyTree."""
  
  stacked = parquet_io.stack_outputs(mock_smc_batch)

  assert isinstance(stacked, SMCOutput)
  
  assert isinstance(stacked.input_config, list)
  assert len(stacked.input_config) == len(mock_smc_batch)
  assert all(isinstance(cfg, SMCConfig) for cfg in stacked.input_config)
  
  assert stacked.input_config[0].seed_sequence == "run_0"
  assert stacked.input_config[1].seed_sequence == "run_1"
  assert stacked.input_config[2].seed_sequence == "run_2"

  batch_size = len(mock_smc_batch)
  num_gens = mock_smc_batch[0].mean_combined_fitness_per_gen.shape[0]

  chex.assert_shape(stacked.mean_combined_fitness_per_gen, (batch_size, num_gens))
  chex.assert_shape(stacked.final_logZhat, (batch_size,))
  chex.assert_shape(stacked.final_amino_acid_entropy, (batch_size,))


def test_stack_outputs_empty_list():
  """Test that stacking an empty list raises a ValueError."""
  with pytest.raises(ValueError, match="Cannot stack an empty list"):
    parquet_io.stack_outputs([])


def test_save_and_load_batch_output(
  temp_dir: Path, mock_smc_batch: list[SMCOutput]
):
  """Test the roundtrip of saving and loading a batch file."""
  # The original test failed because the PyTree definition of SMCOutput
  # was incorrect, causing serialization to fail. With the monkeypatch,
  # this now works as expected.
  batch_file = temp_dir / "test_batch.flax"
  
  parquet_io.save_batch_output(mock_smc_batch, batch_file)
  assert batch_file.exists()

  loaded_batch = parquet_io.load_batch_output(batch_file)
  
  # The loaded object should be equivalent to the original stacked object
  chex.assert_trees_all_close(
    parquet_io.stack_outputs(mock_smc_batch),
    loaded_batch,
    rtol=1e-5,
  )
  assert isinstance(loaded_batch, SMCOutput)


def test_load_nonexistent_file(temp_dir: Path):
  """Test that loading a nonexistent file raises FileNotFoundError."""
  with pytest.raises(FileNotFoundError):
    parquet_io.load_batch_output(temp_dir / "nonexistent.flax")


def test_process_batch_to_parquet(
  temp_dir: Path, mock_smc_batch: list[SMCOutput]
):
  """Test processing a batch file into a Parquet file."""
  batch_file = temp_dir / "test_batch.flax"
  parquet_file = temp_dir / "analytical_data.parquet"
  
  parquet_io.save_batch_output(mock_smc_batch, batch_file)
  parquet_io.process_batch_to_parquet(batch_file, parquet_file)

  assert parquet_file.exists()
  df = pl.read_parquet(parquet_file)

  batch_size = len(mock_smc_batch)
  num_gens = mock_smc_batch[0].beta_per_gen.shape[0]

  assert df.shape[0] == batch_size * num_gens
  assert "experiment_name" in df.columns
  assert df["experiment_name"].n_unique() == batch_size


def test_get_target_type_unknown():
  """Test that an unknown type name raises a TypeError."""
  with pytest.raises(TypeError, match="Unknown output_type name"):
    parquet_io._get_target_type("UnknownType")
