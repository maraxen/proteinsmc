from functools import partial

import jax
import jax.numpy as jnp
from jax import jit
from jaxtyping import PRNGKeyArray

from .utils.types import MPNNModel, ProteinSequence, ScalarBool, ScalarFloat


@partial(jit, static_argnames=('mpnn_model_instance', 'mpnn_model_is_active_static'))
def mpnn_score(
    key: PRNGKeyArray,
    protein_sequence: ProteinSequence,
    mpnn_model_instance: MPNNModel, 
    mpnn_model_is_active_static: ScalarBool
) -> ScalarFloat:
    """
    Calculates MPNN score using the JITted _score method of the MPNN model
    instance and then directly applies the JAX-compatible logic from
    mk_mpnn_model._get_score. This version is designed to be JAX-compatible for
    use within other JITted functions.

    Args:
        key: JAX PRNG key.
        protein_sequence: JAX array of the amino acid sequence (integer
                          encoded according to
                          colabdesign.mpnn.model.aa_order).
        mpnn_model_instance: The initialized and prepped
                             colabdesign.mpnn.model.MPNN object.
        mpnn_model_is_active_static: Boolean flag, static, indicating if the
                                     model instance is ready.

    Returns:
        A JAX scalar representing the MPNN score.
    """
    if not mpnn_model_is_active_static:
        raise ValueError(
            "MPNN model instance is not active. "
            "Ensure it is initialized and prepped correctly."
        )
    inputs_dict = mpnn_model_instance._inputs
    X_input = jnp.asarray(inputs_dict["X"])
    mask_pdb_input = jnp.asarray(inputs_dict["mask"], dtype=jnp.float32)
    residue_idx_input = jnp.asarray(inputs_dict["residue_idx"])
    chain_idx_input = jnp.asarray(inputs_dict["chain_idx"])
    score_kwargs = {'S': protein_sequence}
    if "bias" in inputs_dict:
        score_kwargs['bias'] = jnp.asarray(inputs_dict["bias"])
    if "fix_pos" in inputs_dict:
        score_kwargs['fix_pos'] = jnp.asarray(inputs_dict["fix_pos"])
    internal_output_dict = mpnn_model_instance._score(
        X=X_input,
        mask=mask_pdb_input,
        residue_idx=residue_idx_input,
        chain_idx=chain_idx_input,
        key=key,
        **score_kwargs
    )
    effective_mask_for_scoring = mask_pdb_input
    if "fix_pos" in inputs_dict:
        fix_pos_indices = jnp.asarray(inputs_dict["fix_pos"])
        # Ensure fix_pos_indices is not empty before attempting to use it for
        # indexing
        if fix_pos_indices.shape[0] > 0:
              effective_mask_for_scoring = (
                  effective_mask_for_scoring.at[fix_pos_indices].set(0.0)
              )
    logits_for_scoring = internal_output_dict["logits"][..., :20]
    S_true_one_hot = jax.nn.one_hot(
        protein_sequence, num_classes=21
    )[..., :20]
    log_q = jax.nn.log_softmax(logits_for_scoring, axis=-1)
    score_per_position = -(S_true_one_hot * log_q).sum(axis=-1)
    masked_score_sum = (score_per_position * effective_mask_for_scoring).sum()
    sum_effective_mask = effective_mask_for_scoring.sum()
    final_scalar_score = jnp.where(
        sum_effective_mask > 1e-8,
        masked_score_sum / sum_effective_mask,
        0.0
    )
    final_scalar_score = jnp.array(final_scalar_score, dtype=jnp.float32)
    return final_scalar_score
