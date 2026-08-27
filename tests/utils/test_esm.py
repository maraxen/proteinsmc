import jax.numpy as jnp
import numpy as np

from proteinsmc.utils.constants import (
    ESM_BOS_ID,
    ESM_EOS_ID,
    ESM_SEQUENCE_VOCAB,
    PROTEINMPNN_RESTYPES,
)
from proteinsmc.utils.esm import remap_sequences


def test_remap_sequences():
    """Tests that a sequence is correctly remapped to ESM's vocabulary
    and special tokens are added.
    """
    # ProteinMPNN-ordered integer IDs (0-19), per `remap_sequences`'s documented contract.
    # Under PROTEINMPNN_RESTYPES ("ACDEFGHIKLMNPQRSTVWY") these indices decode to "AFEDCY".
    colab_design_sequence = jnp.array([0, 4, 3, 2, 1, 19], dtype=jnp.int32)  # AFEDCY
    seq_len = len(colab_design_sequence)

    ids = remap_sequences(colab_design_sequence)

    # 1. Check shape
    assert ids.shape == (seq_len + 2,), "Output ID shape is incorrect"

    # 2. Check special tokens
    assert ids[0] == ESM_BOS_ID, "BOS token is missing or incorrect"
    assert ids[-1] == ESM_EOS_ID, "EOS token is missing or incorrect"

    # 3. Check remapped sequence content. "Expected" is derived independently of
    # PROTEINMPNN_TO_ESM_AA_MAP_JAX (the map under test) by looking up each residue
    # character's ESM vocab index directly, mirroring
    # tests/utils/test_constants.py::test_proteinmpnn_to_esm_map_matches_literal_derivation.
    # This is the assertion that would actually catch an alias-collision/permutation bug in
    # the map -- indexing through the map itself, as the previous version of this test did,
    # is tautologically true regardless of whether the map is correct.
    expected_esm_ids = [
        ESM_SEQUENCE_VOCAB.index(PROTEINMPNN_RESTYPES[i])
        for i in colab_design_sequence.tolist()
    ]
    np.testing.assert_array_equal(np.asarray(ids[1:-1]), np.asarray(expected_esm_ids))
