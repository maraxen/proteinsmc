"""MPNN scoring functions."""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from prxteinmpnn.mpnn import get_mpnn_model
from prxteinmpnn.scoring.score import make_score_sequence
from prxteinmpnn.utils.decoding_order import (
    DecodingOrder,
    random_decoding_order,
    single_decoding_order,
)

if TYPE_CHECKING:
    from prxteinmpnn.utils.types import (
        Array,
        AtomMask,
        ChainIndex,
        Float,
        ModelParameters,
        PRNGKeyArray,
        ResidueIndex,
        StructureAtomicCoordinates,
    )

    from proteinsmc.models.fitness import FitnessFn
    from proteinsmc.models.types import ProteinSequence


DEFAULT_MPNN_MODEL = get_mpnn_model()


class DecodingOrderEnum(enum.Enum):
    """Enum for decoding order types."""

    SAME_RANDOM = "same_random"
    KEY_RANDOM = "random_each"
    SEQUENTIAL = "sequential"


def make_mpnn_score(
    mpnn_model_params: ModelParameters,
    coordinates: StructureAtomicCoordinates,
    atom_mask: AtomMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    decoding_order_enum: DecodingOrderEnum,
) -> FitnessFn:
    """Create a scoring function for the MPNN model.

    Args:
        mpnn_model_params (ModelParameters): Parameters for the MPNN model.
        coordinates (StructureAtomicCoordinates): Atomic coordinates of the protein.
        atom_mask (AtomMask): Mask for atoms.
        residue_index (ResidueIndex): Index of each residue.
        chain_index (ChainIndex): Index of the chain for each residue.
        decoding_order_enum (DecodingOrderEnum): Enum specifying the decoding order type.

    Returns:
        A function that scores a protein sequence using the MPNN model.

    """
    input_args = (
        coordinates,
        atom_mask,
        residue_index,
        chain_index,
    )

    match decoding_order_enum:
        case DecodingOrderEnum.KEY_RANDOM:
            score_fn = make_score_sequence(
                model_parameters=mpnn_model_params,
                decoding_order_fn=random_decoding_order,
            )

            @jax.jit
            def mpnn_score(
                key: PRNGKeyArray,
                protein_sequence: ProteinSequence,
                _context: Array | None = None,
            ) -> Float:
                """Scores a protein sequence using the MPNN model."""
                return score_fn(
                    key,
                    protein_sequence,
                    *input_args,
                )[0]

            return mpnn_score

        case DecodingOrderEnum.SAME_RANDOM | DecodingOrderEnum.SEQUENTIAL:
            if decoding_order_enum == DecodingOrderEnum.SEQUENTIAL:

                def sequential_decode_order(
                    _key: PRNGKeyArray,
                    num_residues: int,
                ) -> tuple[DecodingOrder, PRNGKeyArray]:
                    """Generate a sequential decoding order."""
                    return jnp.arange(num_residues, dtype=jnp.int32), _key

                decoding_order_fn = sequential_decode_order
            else:
                decoding_order_fn = single_decoding_order

            score_fn = make_score_sequence(
                model_parameters=mpnn_model_params,
                decoding_order_fn=decoding_order_fn,
            )

            @jax.jit
            def mpnn_score(
                _key: PRNGKeyArray,
                protein_sequence: ProteinSequence,
                _context: Array | None = None,
            ) -> Float:
                """Scores a protein sequence using the MPNN model."""
                return score_fn(
                    _key,
                    protein_sequence,
                    *input_args,
                )[0]

            return mpnn_score

        case _:
            msg = f"Unsupported decoding order enum: {decoding_order_enum}"
            raise ValueError(msg)