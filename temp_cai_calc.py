import jax.numpy as jnp
from src.scoring.cai import cai_score
from src.utils.constants import COLABDESIGN_X_INT
from src.utils.types import ProteinSequence


# test_cai_score_valid_sequence
nuc_seq_valid = jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=jnp.int32)
aa_seq_valid = jnp.array([11, 13, 7], dtype=jnp.int32)
cai_valid = cai_score(nuc_seq_valid, aa_seq_valid)
print(f"cai_valid: {cai_valid}")

# test_cai_score_sequence_with_x_codon
nuc_seq_with_stop = jnp.array([0, 0, 0, 3, 0, 0, 2, 2, 2], dtype=jnp.int32)
aa_seq_with_x = jnp.array([11, COLABDESIGN_X_INT, 7], dtype=jnp.int32)
cai_with_x = cai_score(nuc_seq_with_stop, aa_seq_with_x)
print(f"cai_with_x: {cai_with_x}")

# test_cai_score_with_different_codons
nuc_seq_invalid_codon = jnp.array([0, 0, 0, 3, 3, 3, 2, 2, 2], dtype=jnp.int32)
aa_seq_different = jnp.array([11, 19, 7], dtype=jnp.int32)
cai_different = cai_score(nuc_seq_invalid_codon, aa_seq_different)
print(f"cai_different: {cai_different}")
