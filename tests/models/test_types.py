"""Tests for type aliases and imports."""

from __future__ import annotations

import pytest

from proteinsmc.models.types import EvoSequence, MPNNModel, NucleotideSequence, ProteinSequence


class TestTypeAliases:
  """Test cases for type aliases."""

  def test_nucleotide_sequence_import(self) -> None:
    """Test that NucleotideSequence can be imported."""
    assert NucleotideSequence is not None

  def test_protein_sequence_import(self) -> None:
    """Test that ProteinSequence can be imported."""
    assert ProteinSequence is not None

  def test_evo_sequence_import(self) -> None:
    """Test that EvoSequence can be imported."""
    assert EvoSequence is not None

  def test_mpnn_model_import(self) -> None:
    """Test that MPNNModel can be imported."""
    assert MPNNModel is not None

  def test_evo_sequence_union_type(self) -> None:
    """Test that EvoSequence is a union of NucleotideSequence and ProteinSequence."""
    # This test ensures the type alias is defined correctly
    # In runtime, these are all the same (Int[Array, ...])
    # but the type checker should understand the union
    assert hasattr(EvoSequence, "__args__") or str(EvoSequence).find("|") != -1


class TestMPNNModelIntegration:
  """Test cases for MPNN model integration."""

  def test_mpnn_model_callable(self) -> None:
    """Test that MPNNModel is callable (it's a factory function)."""
    assert callable(MPNNModel)

  @pytest.mark.skip(reason="Requires MPNN dependencies and may be slow")
  def test_mpnn_model_creation(self) -> None:
    """Test creating an MPNN model instance (skipped by default)."""
    # This test is skipped by default as it requires external dependencies
    # and may be slow. It can be enabled for full integration testing.
    model = MPNNModel()
    assert model is not None
