import jax.numpy as jnp
import chex

from proteinsmc.utils.esm import remap_sequences
from proteinsmc.utils.constants import (
    COLABDESIGN_TO_ESM_AA_MAP_JAX,
    ESM_BOS_ID,
    ESM_EOS_ID,
    ESM_PAD_ID,
    ESM_MASK_ID,
)

# --- Tests for utils/esm.py ---

def test_remap_sequences():
    """
    Tests that a sequence is correctly remapped to ESM's vocabulary,
    special tokens are added, and padding is applied correctly.
    """
    # A sample sequence using ColabDesign integer representation (0-19 for AAs)
    colab_design_sequence = jnp.array([0, 4, 3, 2, 1, 19], dtype=jnp.int32) # ARNDCQ
    seq_len = len(colab_design_sequence)

    ids, attention_mask = remap_sequences(colab_design_sequence)

    # 1. Check shapes
    assert ids.shape == (8,), "Padded ID shape is incorrect"
    assert attention_mask.shape == (8,), "Attention mask shape is incorrect"

    # 2. Check special tokens
    assert ids[0] == ESM_BOS_ID, "BOS token is missing or incorrect"
    assert ids[seq_len + 1] == ESM_EOS_ID, "EOS token is missing or incorrect"

    # 3. Check remapped sequence content
    expected_esm_ids = COLABDESIGN_TO_ESM_AA_MAP_JAX[colab_design_sequence]
    assert jnp.array_equal(ids[1:seq_len + 1], expected_esm_ids)

    # 4. Check padding
    assert jnp.all(ids[seq_len + 2:] == ESM_PAD_ID), "Padding is incorrect"

    # 5. Check attention mask
    expected_mask = jnp.concatenate([
        jnp.ones(seq_len + 2, dtype=jnp.int32),
        jnp.zeros(8 - (seq_len + 2), dtype=jnp.int32)
    ])
    assert jnp.array_equal(attention_mask, expected_mask)