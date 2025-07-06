import jax.numpy as jnp
import pytest

# Mock constants and translation functions for isolated testing
from proteinsmc.utils.initiate import generate_template_population


def test_generate_template_population_protein_to_nucleotide():
  # 'MF' -> 'ATGTTT'
  pop = generate_template_population(
    initial_sequence="MF",
    population_size=3,
    input_sequence_type="protein",
    output_sequence_type="nucleotide",
  )
  print(pop)
  expected = jnp.array([[0, 3, 2, 3, 3, 1]] * 3, dtype=jnp.int8)
  assert pop.shape == (3, 6)
  assert jnp.all(pop == expected)


def test_generate_template_population_nucleotide_to_protein():
  # 'ATGTTT' -> 'MF'
  pop = generate_template_population(
    initial_sequence="ATGTTT",
    population_size=2,
    input_sequence_type="nucleotide",
    output_sequence_type="protein",
  )
  expected = jnp.array([[12, 13]] * 2, dtype=jnp.int8)
  assert pop.shape == (2, 2)
  assert jnp.all(pop == expected)


def test_invalid_sequence_type():
  with pytest.raises(ValueError):
    generate_template_population(
      initial_sequence="ATG",
      population_size=1,
      input_sequence_type="rna",  # type: ignore[unused-variable, assignment]  # noqa: E501
      output_sequence_type="nucleotide",
    )


def test_invalid_output_sequence_type():
  with pytest.raises(ValueError):
    generate_template_population(
      initial_sequence="ATG",
      population_size=1,
      input_sequence_type="nucleotide",
      output_sequence_type="rna",  # type: ignore[unused-variable, assignment]  # noqa: E501
    )


def test_invalid_amino_acid():
  # 'Z' is not in RES_TO_CODON_CHAR
  with pytest.raises(ValueError):
    generate_template_population(
      initial_sequence="Z",
      population_size=1,
      input_sequence_type="protein",
      output_sequence_type="nucleotide",
    )
