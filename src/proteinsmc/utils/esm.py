"""Utilities for remapping amino acid sequences from ColabDesign to ESM format.

Contains code adapted from the esmj library:
https://github.com/escalante-bio/esmj
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

import einops
import equinox as eqx
import huggingface_hub
import jax
import numpy as np
from equinox import field
from jax import numpy as jnp
from jaxtyping import Array, Float, Int

if TYPE_CHECKING:
  from collections.abc import Callable

  from jaxtyping import PRNGKeyArray

  from proteinsmc.models.types import ProteinSequence


from .constants import (
  ESM_BOS_ID,
  ESM_EOS_ID,
  PROTEINMPNN_TO_ESM_AA_MAP_JAX,
)


@jax.jit
def remap_sequences(
  sequence: ProteinSequence,
) -> ProteinSequence:
  """Remap amino acid integer IDs from ProteinMPNN's scheme to ESM's token scheme.

  Also add BOS/EOS tokens and pad the sequence.

  Args:
      sequence: A ProteinSequence containing amino acid integer IDs.

  Returns:
      A tuple of (esm_amino_acid_ids, attention_mask) as JAX arrays.

  """
  esm_aa_ints_raw = PROTEINMPNN_TO_ESM_AA_MAP_JAX[sequence]

  return jnp.concatenate(
    [
      jnp.array([ESM_BOS_ID], dtype=jnp.int32),
      esm_aa_ints_raw,
      jnp.array([ESM_EOS_ID], dtype=jnp.int32),
    ],
  )


class AbstractFromTorch(eqx.Module):
  """Abstract base class for modules converted from PyTorch."""


def _vmap(f: Callable, tensor: Array, *args: Any) -> Array:  # noqa: ANN401
  """Apply vmap to all leading dimensions of a tensor."""
  for _ in range(len(tensor.shape) - 1):
    f = jax.vmap(f)
  return f(tensor, *args)


def vmap_to_last_dimension(f: Callable) -> Callable:
  """Wrap a function to apply vmap to all leading dimensions of its first argument."""
  return partial(_vmap, f)


LinearWeights = Float[Array, "Out In"]
LinearBias = Float[Array, "Out"]
LinearInput = Float[Array, "... In"]
LinearOutput = Float[Array, "... Out"]


class Linear(AbstractFromTorch):
  """Linear layer that matches PyTorch semantics."""

  weight: LinearWeights
  bias: LinearBias | None

  def __call__(self, x: LinearInput) -> LinearOutput:
    """Apply the linear transformation."""
    o = einops.einsum(x, self.weight, "... In, Out In -> ... Out")
    if self.bias is not None:
      o = o + jnp.broadcast_to(self.bias, x.shape[:-1] + (self.bias.shape[-1],))
    return o


LayerNormWeight = Float[Array, "Dim"]
LayerNormBias = Float[Array, "Dim"]
LayerNormInput = Float[Array, "... Dim"]
LayerNormOutput = Float[Array, "... Dim"]


class LayerNorm(AbstractFromTorch):
  """LayerNorm that matches PyTorch semantics."""

  weight: LayerNormWeight | None
  bias: LayerNormBias | None
  eps: float = field(static=True)

  def __call__(self, x: LayerNormInput) -> LayerNormOutput:
    """Apply layer normalization."""
    ln = eqx.nn.LayerNorm(
      shape=x.shape[-1],
      eps=self.eps,
      use_weight=self.weight is not None,
      use_bias=self.bias is not None,
    )
    ln = eqx.tree_at(
      lambda layer: (layer.weight, layer.bias),
      ln,
      (self.weight, self.bias),
      is_leaf=lambda n: n is None,
    )
    return vmap_to_last_dimension(ln)(x)


class Sequential(AbstractFromTorch):
  """A sequence of modules, matching PyTorch's Sequential."""

  _modules: dict[str, AbstractFromTorch]

  def __call__(self, x: Array) -> Array:
    """Apply each module in sequence."""
    for idx in range(len(self._modules)):
      module = self._modules[str(idx)]
      if module is not None:
        x = module(x)  # type: ignore[call-issue]
    return x


SparseEmbeddingInput = Int[Array, "*Batch"]
SparseEmbeddingOutput = Float[Array, "*Batch Dim"]


class SparseEmbedding(AbstractFromTorch):
  """Wrapper for eqx.nn.Embedding to handle batched inputs."""

  embedding: eqx.nn.Embedding

  def __call__(self, indices: SparseEmbeddingInput) -> SparseEmbeddingOutput:
    """Apply the embedding to batched indices."""
    f = self.embedding
    for _ in range(indices.ndim):
      f = jax.vmap(f)
    return f(indices)


SwiGLUOutput = Float[Array, "... D"]


def swiglu(x: Array) -> SwiGLUOutput:
  """SwiGLU activation function."""
  x1, x2 = jnp.split(x, 2, axis=-1)
  return jax.nn.silu(x1) * x2


InverseFrequencyOutput = Float[Array, "D"]
RotateHalfIO = Float[Array, "... D"]


class RotaryEmbedding(AbstractFromTorch):
  """Rotary Positional Embedding (RoPE)."""

  dim: int = field(static=True)
  base: int = field(static=True, default=10000)

  @property
  def inverse_freq(self) -> InverseFrequencyOutput:
    """Compute the inverse frequency for RoPE."""
    return 1.0 / (self.base ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))

  @staticmethod
  def rotate_half(x: RotateHalfIO) -> RotateHalfIO:
    """Rotate the last dimension of the input tensor by half."""
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)

  @staticmethod
  def apply_rotary_emb(x: Array, cos: Array, sin: Array) -> Array:
    """Apply rotary embeddings to the input tensor."""
    ro_dim = cos.shape[-1] * 2
    seqlen = x.shape[1]
    cos, sin = cos[:seqlen], sin[:seqlen]
    cos = einops.repeat(cos, "s d -> s 1 (2 d)")
    sin = einops.repeat(sin, "s d -> s 1 (2 d)")
    x_rot, x_pass = x[..., :ro_dim], x[..., ro_dim:]
    x_rotated = (x_rot * cos) + (RotaryEmbedding.rotate_half(x_rot) * sin)
    return jnp.concatenate([x_rotated, x_pass], axis=-1)

  def __call__(self, q: Array, k: Array) -> tuple[Array, Array]:
    """Apply RoPE to query and key tensors."""
    n = q.shape[1]
    t = jnp.arange(n, dtype=jnp.float32)
    freqs = jnp.outer(t, self.inverse_freq)
    cos, sin = jnp.cos(freqs), jnp.sin(freqs)
    return self.apply_rotary_emb(q, cos, sin), self.apply_rotary_emb(k, cos, sin)


InternalProjection = Float[Array, "B N D"]


class MultiHeadAttention(AbstractFromTorch):
  """Multi-head attention layer with RoPE."""

  d_model: int = field(static=True)
  n_heads: int = field(static=True)
  d_head: int = field(static=True)
  layernorm_qkv: Sequential
  out_proj: Linear
  rotary: RotaryEmbedding
  q_ln: LayerNorm
  k_ln: LayerNorm

  def _apply_rotary(self, q: Array, k: Array) -> tuple[Array, Array]:
    q = einops.rearrange(q, "b s (h d) -> b s h d", h=self.n_heads)
    k = einops.rearrange(k, "b s (h d) -> b s h d", h=self.n_heads)
    q, k = self.rotary(q, k)
    return (
      einops.rearrange(q, "b s h d -> b s (h d)"),
      einops.rearrange(k, "b s h d -> b s (h d)"),
    )

  def __call__(self, x: InternalProjection) -> InternalProjection:
    """Apply multi-head attention with RoPE."""
    qkv = self.layernorm_qkv(x)
    query, key, value = jnp.split(qkv, 3, axis=-1)
    query, key = self.q_ln(query), self.k_ln(key)
    query, key = self._apply_rotary(query, key)
    query, key, value = (
      einops.rearrange(t, "b s (h d) -> b h s d", h=self.n_heads) for t in (query, key, value)
    )
    context = jax.nn.dot_product_attention(query, key, value)
    return self.out_proj(einops.rearrange(context, "b h s d -> b s (h d)"))


class UnifiedTransformerBlock(AbstractFromTorch):
  """A transformer block with attention and feedforward network."""

  ffn: Sequential
  attn: MultiHeadAttention
  scaling_factor: float = field(static=True)

  def __call__(self, x: InternalProjection) -> InternalProjection:
    """Apply a transformer block with attention and feedforward network."""
    x = x + self.attn(x) / self.scaling_factor
    return x + self.ffn(x) / self.scaling_factor


class TransformerStack(AbstractFromTorch):
  """A stack of transformer blocks."""

  block_params: UnifiedTransformerBlock
  block_static: UnifiedTransformerBlock
  norm: LayerNorm

  def __call__(self, x: InternalProjection) -> tuple[InternalProjection, InternalProjection]:
    """Apply a stack of transformer blocks."""

    def body(
      carry: InternalProjection,
      params: Float,
    ) -> tuple[InternalProjection, InternalProjection]:
      layer = eqx.combine(params, self.block_static)
      output = layer(carry)
      return output, output

    final, all_states = jax.lax.scan(body, x, self.block_params)
    return self.norm(final), all_states


Logits = Float[Array, "B N V"]
Embeddings = Float[Array, "B N D"]
Hiddens = Float[Array, "L B N D"]


class ESMCOutput(eqx.Module):
  """Output structure for the ESMC model."""

  logits: Logits
  embedding: Embeddings
  hiddens: Hiddens


Tokens = Int[Array, "B N"]


class ESMC(AbstractFromTorch):
  """Equinox implementation of the ESM model."""

  embed: SparseEmbedding
  transformer: TransformerStack
  sequence_head: Sequential
  vocab: dict[str, int] = field(static=True)

  def __call__(self, tokens: Tokens) -> ESMCOutput:
    """Forward pass of the ESMC model."""
    x = self.embed(tokens)
    x, hiddens = self.transformer(x)
    logits = self.sequence_head(x)
    return ESMCOutput(logits=logits, embedding=x, hiddens=hiddens)

  def tokenize(self, sequence: str) -> np.ndarray:
    """Tokenize a protein sequence into ESM integer IDs."""
    return np.array(
      [self.vocab["<cls>"]]
      + [self.vocab.get(c, self.vocab["<unk>"]) for c in sequence.upper()]
      + [self.vocab["<eos>"]],
      dtype=np.int32,
    )


MODEL_CONFIGS = {
  "esmc_600m": {
    "vocab_size": 64,
    "embed_dim": 1152,
    "n_layers": 36,
    "n_heads": 18,
  },
  "esmc_300m": {
    "vocab_size": 64,
    "embed_dim": 960,
    "n_layers": 30,
    "n_heads": 15,
  },
}


def save_model(filename: str, model: ESMC, hyperparams: dict) -> None:
  """Save hyperparameters and model weights to a single file."""
  with Path(filename).open("wb") as f:
    hyperparam_str = json.dumps(hyperparams)
    f.write((hyperparam_str + "\n").encode())

    eqx.tree_serialise_leaves(f, model)


def load_model(model_name: str, key: PRNGKeyArray) -> ESMC:
  """Load a model from a file containing hyperparameters and weights."""
  if not Path(f"esm_models/{model_name}.eqx").exists():
    huggingface_hub.hf_hub_download(
      repo_id="maraxen/esmc_models",
      filename=f"{model_name}.eqx",
      local_dir="esm_models",
      local_dir_use_symlinks=False,
    )
  with Path(f"esm_models/{model_name}.eqx").open("rb") as f:
    hyperparams = json.loads(f.readline().decode())
    model_name = hyperparams.get("model_name")
    if not model_name:
      msg = "Hyperparameters in file must contain a 'model_name' key."
      raise ValueError(msg)

    skeleton = create_esmc_skeleton(key, model_name)

    return eqx.tree_deserialise_leaves(f, skeleton)


def robust_deserialise_filter_spec(f: BinaryIO, leaf: Any) -> Any:  # noqa: ANN401
  """Force loaded arrays to match the shape and dtype of the template leaf."""
  if isinstance(leaf, (jax.Array, np.ndarray)):
    loaded_leaf = jnp.load(f, allow_pickle=False)
    return jnp.broadcast_to(loaded_leaf, leaf.shape).astype(leaf.dtype)
  return eqx.default_deserialise_filter_spec(f, leaf)


def create_esmc_skeleton(key: PRNGKeyArray, model_name: str) -> ESMC:
  """Create an uninitialized ESMC model skeleton for deserialization."""
  if model_name not in MODEL_CONFIGS:
    msg = f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}"
    raise ValueError(msg)

  cfg = MODEL_CONFIGS[model_name]
  vocab_size, embed_dim, n_layers, n_heads = cfg.values()

  keys = jax.random.split(key, 20)
  d_head = embed_dim // n_heads

  if model_name == "esmc_600m":
    ffn_embed_dim = 6144
  elif model_name == "esmc_300m":
    ffn_embed_dim = 5120
  else:
    ffn_embed_dim = embed_dim * 4

  eps = 1e-5

  def _create_block(k: PRNGKeyArray) -> UnifiedTransformerBlock:
    k_attn, k_ffn, _k_ln = jax.random.split(k, 3)

    # --- Attention Block ---
    k_qkv_w, k_out_w = jax.random.split(k_attn, 2)
    layernorm_qkv = Sequential(
      {
        "0": LayerNorm(jnp.ones(embed_dim), jnp.zeros(embed_dim), eps),
        "1": Linear(jax.random.normal(k_qkv_w, (3 * embed_dim, embed_dim)), bias=None),
      },
    )
    out_proj = Linear(jax.random.normal(k_out_w, (embed_dim, embed_dim)), bias=None)
    attn = MultiHeadAttention(
      d_model=embed_dim,
      n_heads=n_heads,
      d_head=d_head,
      layernorm_qkv=layernorm_qkv,
      out_proj=out_proj,
      rotary=RotaryEmbedding(dim=d_head),
      q_ln=LayerNorm(weight=jnp.ones(embed_dim), bias=None, eps=eps),
      k_ln=LayerNorm(weight=jnp.ones(embed_dim), bias=None, eps=eps),
    )

    # --- FFN Block ---
    k_l1_w, k_l2_w = jax.random.split(k_ffn, 2)
    ffn = Sequential(
      {
        "0": LayerNorm(jnp.ones(embed_dim), jnp.zeros(embed_dim), eps),
        "1": Linear(jax.random.normal(k_l1_w, (ffn_embed_dim, embed_dim)), bias=None),
        "2": eqx.nn.Lambda(swiglu),  # type: ignore[arg-type]
        "3": Linear(jax.random.normal(k_l2_w, (embed_dim, ffn_embed_dim // 2)), bias=None),
      },
    )

    return UnifiedTransformerBlock(ffn, attn, scaling_factor=np.sqrt(n_layers))

  embed = SparseEmbedding(eqx.nn.Embedding(vocab_size, embed_dim, key=keys[0]))

  # --- Transformer Stack ---
  single_block = _create_block(keys[1])
  params, static = eqx.partition(single_block, eqx.is_inexact_array)
  stacked_params = jax.tree_util.tree_map(lambda x: jnp.stack([x] * n_layers), params)

  final_norm = LayerNorm(weight=jnp.ones(embed_dim), bias=None, eps=eps)
  transformer = TransformerStack(stacked_params, static, final_norm)

  # --- Sequence Head ---
  k_h1_w, k_h1_b, k_h2_w, k_h2_b = jax.random.split(keys[2], 4)
  sequence_head = Sequential(
    {
      "0": Linear(
        jax.random.normal(k_h1_w, (embed_dim, embed_dim)),
        jax.random.normal(k_h1_b, (embed_dim,)),
      ),
      "1": eqx.nn.Lambda(jax.nn.gelu),  # type: ignore[arg-type]
      "2": LayerNorm(jnp.ones(embed_dim), jnp.zeros(embed_dim), eps),
      "3": Linear(
        jax.random.normal(k_h2_w, (vocab_size, embed_dim)),
        jax.random.normal(k_h2_b, (vocab_size,)),
      ),
    },
  )

  vocab_list = [
    "<cls>",
    "<pad>",
    "<eos>",
    "<unk>",
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "X",
    "B",
    "U",
    "Z",
    "O",
    ".",
    "-",
    "|",
    "<mask>",
  ]
  vocab = {c: i for i, c in enumerate(vocab_list)}

  return ESMC(embed, transformer, sequence_head, vocab)
