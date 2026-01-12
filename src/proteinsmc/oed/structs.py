"""Dataclasses for OED."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from flax.struct import dataclass
from jaxtyping import Float, Int


@dataclass
class OEDDesign:
  """Dataclass representing an OED design configuration."""

  N: Int  # Sequence length
  K: Int  # Number of interacting sites
  q: Int  # Alphabet size
  population_size: Int  # Size of the evolving population
  n_generations: Int  # Number of generations to simulate
  mutation_rate: Float  # Mutation rate per site
  diversification_ratio: Float  # Ratio of diversification in the population

  def replace(self, **kwargs: Any) -> OEDDesign:  # noqa: ANN401
    """Create a new instance with updated fields."""
    return replace(self, **kwargs)


@dataclass
class OEDPredictedVariables:
  """Dataclass representing predicted variables from OED simulations."""

  information_gain: Float  # Predicted information gain
  barrier_crossing_frequency: Float  # Frequency of barrier crossings
  final_sequence_entropy: Float  # Entropy of the final sequence distribution
  jsd_from_original_population: Float  # Jensen-Shannon divergence from original population
  geometric_fitness_mean: Float  # Geometric mean of fitness over generations
