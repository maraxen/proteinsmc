"""Registry for fitness functions and their configurations."""

from __future__ import annotations

from proteinsmc.models.fitness import CombineRegistry, FitnessRegistry
from proteinsmc.scoring.cai import CAI_FITNESS
from proteinsmc.scoring.mpnn import MPNN_FITNESS

from .combine import SUM_COMBINE, WEIGHTED_COMBINE

FITNESS_REGISTRY = FitnessRegistry(
  items={
    "mpnn_score": MPNN_FITNESS,
    "cai_score": CAI_FITNESS,
  },
)


COMBINE_REGISTRY = CombineRegistry(
  items={
    "sum_combine": SUM_COMBINE,
    "weighted_combine": WEIGHTED_COMBINE,
  },
)
