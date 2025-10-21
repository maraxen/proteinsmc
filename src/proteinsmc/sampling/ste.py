"""Straight-Through Estimator (STE) for JAX.

Note: Only use this for discrete optimization problems where you want to allow gradients
to pass through the argmax operation. Useful for tasks like protein sequence
optimization when a model outputs logits for amino acid sequences.

Unclear if the optimized sequences will be valid proteins, so this is a heuristic
approach to allow gradient-based optimization on discrete outputs to assess how well
other samplers are navigating model landscapes.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
from jaxtyping import Array, Bool, Float

Mask = Bool[Array, "sequence_length"]
Logits = Float[Array, "sequence_length num_classes"]
CEELoss = Float[Array, ""]
STELoopFn = Callable[[Logits, Logits, Mask], Logits]
"""Type for the STE optimization loop function."""


def straight_through_estimator(logits: Logits) -> Logits:
  """Implement the straight-through estimator (STE).

  Allow gradients to pass through the discrete argmax operation.
  """
  probs = jax.nn.softmax(logits, axis=-1)
  one_hot = jax.nn.one_hot(jnp.argmax(probs, axis=-1), num_classes=probs.shape[-1])
  return jax.lax.stop_gradient(one_hot - probs) + probs


DEFAULT_LEARNING_RATE = 1e-2
DEFAULT_NUM_STEPS = 200


def create_pure_optimization_fn(
  learning_rate: float = DEFAULT_LEARNING_RATE,
  num_steps: int = DEFAULT_NUM_STEPS,
) -> STELoopFn:
  """Create a pure, JIT-compiled function to run STE optimization."""

  def get_loss(logits_to_optimize: Logits, target_logits: Logits, mask: Mask) -> CEELoss:
    """Calculate cross-entropy between one-hot sequence (from STE) and target distribution.

    Args:
        logits_to_optimize: Logits to optimize, shape (sequence_length, num_classes).
        target_logits: Target logits for the sequence, shape (sequence_length, num_classes).
          These are the logits from the model that we want to match, such as MPNN model's
          unconditional logits.
        mask: Boolean mask indicating valid positions in the sequence.
          Used to ignore padding or invalid positions.

    Returns:
        Loss value as a scalar.

    Example:
        >>> logits_to_optimize = jnp.array([[0.1, 0.9], [0.8, 0.2]])
        >>> target_logits = jnp.array([[0.2, 0.8], [0.7, 0.3]])
        >>> mask = jnp.array([True, True])
        >>> loss = get_loss(logits_to_optimize, target_logits, mask)

    """
    seq_one_hot = straight_through_estimator(logits_to_optimize)
    target_log_probs = jax.nn.log_softmax(target_logits)
    loss_per_position = -(seq_one_hot * target_log_probs).sum(axis=-1)
    return (loss_per_position * mask).sum() / (mask.sum() + 1e-8)

  @jax.jit
  def run_optimization_loop(
    initial_logits: Logits,
    target_logits: Logits,
    mask: Mask,
  ) -> Logits:
    """Run the full optimization loop for a SINGLE protein."""
    opt_init, opt_update, get_params = optimizers.adam(learning_rate)
    loss_and_grad_fn = jax.value_and_grad(get_loss)

    def step_fn(i: int, opt_state: optimizers.OptimizerState) -> optimizers.OptimizerState:
      current_logits = get_params(opt_state)
      _, grad = loss_and_grad_fn(current_logits, target_logits, mask)
      return opt_update(i, grad, opt_state)

    opt_state_initial = opt_init(initial_logits)
    opt_state_final = jax.lax.fori_loop(0, num_steps, step_fn, opt_state_initial)

    return get_params(opt_state_final)

  return jax.vmap(run_optimization_loop, in_axes=(0, 0, 0))
