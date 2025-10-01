import jax.numpy as jnp
import numpy as np
import chex

from proteinsmc.utils.esm import remap_sequences
from proteinsmc.utils.constants import (
    PROTEINMPNN_TO_ESM_AA_MAP_JAX,
    ESM_BOS_ID,
    ESM_EOS_ID,
)

def test_remap_sequences():
    """
    Tests that a sequence is correctly remapped to ESM's vocabulary,
    and special tokens are added.
    """
    # A sample sequence using ColabDesign integer representation (0-19 for AAs)
    colab_design_sequence = jnp.array([0, 4, 3, 2, 1, 19], dtype=jnp.int32) # ARNDCQ
    seq_len = len(colab_design_sequence)

    ids = remap_sequences(colab_design_sequence)

    # 1. Check shape
    assert ids.shape == (seq_len + 2,), "ID shape is incorrect"

    # 2. Check special tokens
    assert ids[0] == ESM_BOS_ID, "BOS token is missing or incorrect"
    assert ids[seq_len + 1] == ESM_EOS_ID, "EOS token is missing or incorrect"

    # 3. Check remapped sequence content
    expected_esm_ids = PROTEINMPNN_TO_ESM_AA_MAP_JAX[colab_design_sequence]
    np.testing.assert_array_equal(np.asarray(ids[1:seq_len + 1]), np.asarray(expected_esm_ids))