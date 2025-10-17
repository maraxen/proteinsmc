"""Utility functions for protein sequence modeling and sampling."""

from .annealing import (
  ANNEALING_REGISTRY,
  cosine_schedule,
  exponential_schedule,
  linear_schedule,
  static_schedule,
)
from .constants import (
  AA_CHAR_TO_INT_MAP,
  CODON_INT_TO_RES_INT_JAX,
  CODON_TO_RES_CHAR,
  ECOLI_CODON_FREQ_CHAR,
  ECOLI_CODON_FREQ_JAX,
  ECOLI_MAX_FREQS_JAX,
  INT_TO_AA_CHAR_MAP,
  INT_TO_NUCLEOTIDES_CHAR_MAP,
  MAX_NUC_INT,
  NUCLEOTIDES_CHAR,
  NUCLEOTIDES_INT_MAP,
  NUCLEOTIDES_JAX,
  PROTEINMPNN_X_INT,
  RES_TO_CODON_CHAR,
  STOP_INT,
  UNKNOWN_AA_INT,
  ECOLI_MAX_FREQS_JAX_list,
)
from .fitness import FITNESS_FUNCTIONS, get_fitness_function
from .initiate import generate_template_population
from .jax_utils import chunked_map
from .memory import (
  BenchmarkResult,
  auto_tune_chunk_size,
  create_test_population,
  estimate_memory_usage,
  get_device_memory_mb,
  suggest_chunk_size_heuristic,
)
from .metrics import (
  calculate_logZ_increment,
  calculate_position_entropy,
  safe_weighted_mean,
  shannon_entropy,
)
from .mutation import (
  _revert_x_codons_if_mutated,
  chunked_mutation_step,
  diversify_initial_sequences,
  mutate,
)
from .pmap_utils import distribute
from .translation import aa_to_nucleotide, nucleotide_to_aa

__all__ = [
  "AA_CHAR_TO_INT_MAP",
  "CODON_INT_TO_RES_INT_JAX",
  "PROTEINMPNN_X_INT",
  "ECOLI_CODON_FREQ_CHAR",
  "ECOLI_CODON_FREQ_JAX",
  "ECOLI_MAX_FREQS_JAX",
  "INT_TO_AA_CHAR_MAP",
  "INT_TO_NUCLEOTIDES_CHAR_MAP",
  "MAX_NUC_INT",
  "NUCLEOTIDES_CHAR",
  "NUCLEOTIDES_INT_MAP",
  "NUCLEOTIDES_JAX",
  "STOP_INT",
  "UNKNOWN_AA_INT",
  "ECOLI_MAX_FREQS_JAX_list",
  "cosine_schedule",
  "exponential_schedule",
  "linear_schedule",
  "static_schedule",
  "BenchmarkResult",
  "auto_tune_chunk_size",
  "create_test_population",
  "estimate_memory_usage",
  "get_device_memory_mb",
  "suggest_chunk_size_heuristic",
  "calculate_position_entropy",
  "shannon_entropy",
  "_revert_x_codons_if_mutated",
  "diversify_initial_sequences",
  "mutate",
  "nucleotide_to_aa",
  "aa_to_nucleotide",
  "RES_TO_CODON_CHAR",
  "CODON_TO_RES_CHAR",
  "calculate_logZ_increment",
  "generate_template_population",
  "distribute",
  "chunked_map",
  "safe_weighted_mean",
  "chunked_mutation_step",
  "get_fitness_function",
  "FITNESS_FUNCTIONS",
  "ANNEALING_REGISTRY",
]
