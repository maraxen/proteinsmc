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
  auto_tune_batch_size,
  create_test_population,
  estimate_memory_usage,
  get_device_memory_mb,
  suggest_batch_size_heuristic,
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
  "ANNEALING_REGISTRY",
  "CODON_INT_TO_RES_INT_JAX",
  "CODON_TO_RES_CHAR",
  "ECOLI_CODON_FREQ_CHAR",
  "ECOLI_CODON_FREQ_JAX",
  "ECOLI_MAX_FREQS_JAX",
  "FITNESS_FUNCTIONS",
  "INT_TO_AA_CHAR_MAP",
  "INT_TO_NUCLEOTIDES_CHAR_MAP",
  "MAX_NUC_INT",
  "NUCLEOTIDES_CHAR",
  "NUCLEOTIDES_INT_MAP",
  "NUCLEOTIDES_JAX",
  "PROTEINMPNN_X_INT",
  "RES_TO_CODON_CHAR",
  "STOP_INT",
  "UNKNOWN_AA_INT",
  "BenchmarkResult",
  "ECOLI_MAX_FREQS_JAX_list",
  "_revert_x_codons_if_mutated",
  "aa_to_nucleotide",
  "auto_tune_batch_size",
  "calculate_logZ_increment",
  "calculate_position_entropy",
  "chunked_map",
  "chunked_mutation_step",
  "cosine_schedule",
  "create_test_population",
  "distribute",
  "diversify_initial_sequences",
  "estimate_memory_usage",
  "exponential_schedule",
  "generate_template_population",
  "get_device_memory_mb",
  "get_fitness_function",
  "linear_schedule",
  "mutate",
  "nucleotide_to_aa",
  "safe_weighted_mean",
  "shannon_entropy",
  "static_schedule",
  "suggest_batch_size_heuristic",
]
