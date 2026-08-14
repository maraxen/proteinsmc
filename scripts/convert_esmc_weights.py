"""Convert official ESM-C PyTorch weights into the Equinox `.eqx` checkpoint this repo loads.

WHY THIS EXISTS

The `.eqx` checkpoints previously distributed via HuggingFace `maraxen/esmc_models` contain
**randomly-initialised weights**, not converted ESM-C weights. The structure is correct --
right config, right parameter count, right shapes -- so nothing raises, and
`utils.esm.load_model` returns a model that scores every protein as noise. Diagnosed
2026-08-13: on human ubiquitin the model recovered the correct residue at 0 of 76 positions,
per-site log-likelihood was about -75 (expected -1 to -3), and every weight matrix had
standard deviation 1.0 (a trained transformer sits near 0.02-0.05).

This script rebuilds the checkpoint from the official release and refuses to write one that
fails an acceptance test, so the failure cannot recur silently.

THE MAPPING

`create_esmc_skeleton` produces a PyTree whose leaf paths correspond 1:1 with the official
state-dict keys; per-block parameters are stacked along a leading axis in block order.

    .embed.embedding.weight                                  <- embed.weight
    .transformer.block_params.ffn._modules['0'].{weight,bias} <- transformer.blocks.{i}.ffn.0.*
    .transformer.block_params.ffn._modules['1'].weight        <- transformer.blocks.{i}.ffn.1.weight
    .transformer.block_params.ffn._modules['3'].weight        <- transformer.blocks.{i}.ffn.3.weight
    ...attn.layernorm_qkv._modules['0'].{weight,bias}         <- ...attn.layernorm_qkv.0.*
    ...attn.layernorm_qkv._modules['1'].weight                <- ...attn.layernorm_qkv.1.weight
    ...attn.{out_proj,q_ln,k_ln}.weight                       <- ...attn.{out_proj,q_ln,k_ln}.weight
    .transformer.norm.weight                                  <- transformer.norm.weight
    .sequence_head._modules['{0,2,3}'].{weight,bias}          <- sequence_head.{0,2,3}.*

Rather than trusting that table, the script derives each leaf's torch key from its own path
string and asserts every shape matches before writing.

USAGE

    # torch is required; asr's environment has it, this repo's does not
    cd ~/projects/asr && uv run --no-sync python \
        ~/projects/proteinsmc/scripts/convert_esmc_weights.py \
        --torch-weights "$W" --model-name esmc_300m --out esm_models/esmc_300m.eqx

    # where $W is the official release, e.g. under
    #   ~/.cache/huggingface/hub/models--EvolutionaryScale--esmc-300m-2024-12/
    #     snapshots/<rev>/data/weights/esmc_300m_2024_12_v0.pth
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from proteinsmc.utils import constants as C  # noqa: N812
from proteinsmc.utils.esm import MODEL_CONFIGS, create_esmc_skeleton

logger = logging.getLogger("convert_esmc")

# A real protein with a well-known sequence: human ubiquitin (P0CG48), 76 residues.
UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

MATRIX_NDIM = 2
"""Leaves with at least this rank are treated as weight matrices."""

MIN_RECOVERY = 0.30
"""Fraction of positions where argmax must equal the input. A trained ESM-C comfortably
exceeds this -- on a correct port it scores 1.000 on ubiquitin, since the input is unmasked
-- while the untrained checkpoint scored 0.000. The threshold stays loose at 0.30 because it
is meant to catch wholesale randomness, and because a *port* defect (not a weight defect)
can depress it substantially without zeroing it: with the residual-scaling and attention
bugs still present, correctly converted weights measured 0.737 here and still passed. That
is the intended behaviour -- this gate certifies the weights, not the forward pass. Numeric
parity against the reference implementation is what certifies the port."""

MAX_MEDIAN_WEIGHT_STD = 0.75
"""Upper bound on the MEDIAN standard deviation across the model's >=2D leaves.

This is the SECONDARY gate and it is weak. Read `MIN_RECOVERY` as the real one.

Median, not maximum: a trained ESM-C legitimately contains high-variance tensors (on the
official release, per-tensor std runs 0.023 to 1.279 with a median of 0.070 across 123
matrices), so a max-based bound rejects correct weights -- it did, on the first run here,
at 1.954.

The threshold is loose because the quantity is measured on the *stacked* Equinox leaves,
not per torch tensor. Stacking folds 30 blocks into one array and mixes layernorm gains
(which sit near 1.0 even when trained) in with projection matrices, so the trained model
measures 0.476 here rather than 0.070. Against the untrained checkpoint's 1.000 that still
separates, but only by a factor of two -- hence 0.75 rather than a tight bound, so a
different model size cannot fail spuriously. Do not read a pass here as strong evidence;
it only rules out a wholesale random-initialisation."""


def path_to_torch_key(path_str: str) -> str:
  """Translate an Equinox leaf path into its official state-dict key.

  `.transformer.block_params.X` denotes a per-block parameter stacked over the leading
  axis; it becomes `transformer.blocks.{i}.X` and is handled by the caller. Everything
  else maps directly once `._modules['k']` is rewritten to `.k`.
  """
  key = path_str.lstrip(".")
  key = re.sub(r"_modules\['(\w+)'\]", r"\1", key)
  key = key.replace("..", ".")
  # the skeleton wraps the embedding table one level deeper than torch does
  return key.replace("embed.embedding.weight", "embed.weight")


def build_leaf(
  torch_key: str,
  shape: tuple[int, ...],
  state: dict,
  n_layers: int,
) -> jnp.ndarray:
  """Return the array for one skeleton leaf, stacking per-block tensors where needed."""
  if torch_key.startswith("transformer.block_params."):
    suffix = torch_key[len("transformer.block_params.") :]
    stack = [
      jnp.asarray(state[f"transformer.blocks.{i}.{suffix}"].float().numpy())
      for i in range(n_layers)
    ]
    arr = jnp.stack(stack, axis=0)
  else:
    if torch_key not in state:
      msg = f"state dict has no key {torch_key!r}"
      raise KeyError(msg)
    arr = jnp.asarray(state[torch_key].float().numpy())

  if arr.shape != shape:
    msg = f"shape mismatch for {torch_key!r}: skeleton {shape}, weights {arr.shape}"
    raise ValueError(msg)
  return arr


def acceptance_test(model) -> tuple[float, float]:  # noqa: ANN001
  """Score ubiquitin and return (argmax recovery, median weight-matrix std).

  Both are cheap invariants that the untrained checkpoint fails unambiguously: it scored
  0.000 recovery against 1.000 for correctly converted weights on a correct port, and its
  median matrix std was 1.000 against 0.070.

  Note on the recovery figure: this is *unmasked* argmax agreement -- the model sees the
  residue it is scoring -- so a working model should sit very near 1.0, and ubiquitin in
  particular is in every training set. The 0.737 recorded during the original conversion was
  measured while the port still carried the residual-scaling and attention-layout defects
  (fixed in 01c1b08); it is a degraded number, not the reference. Do not read 0.737 as the
  expected value.
  """
  ids = jnp.array([C.PROTEINMPNN_RESTYPES.index(c) for c in UBIQUITIN], dtype=jnp.int32)
  tokens = jnp.concatenate(
    [
      jnp.array([C.ESM_BOS_ID], dtype=jnp.int32),
      C.PROTEINMPNN_TO_ESM_AA_MAP_JAX[ids],
      jnp.array([C.ESM_EOS_ID], dtype=jnp.int32),
    ],
  )
  logits = model(tokens[None, :]).logits[0]
  recovery = float((logits[1:-1].argmax(-1) == tokens[1:-1]).mean())

  mats = [
    leaf
    for leaf in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    if leaf.ndim >= MATRIX_NDIM
  ]
  median_std = float(np.median([float(m.std()) for m in mats]))
  return recovery, median_std


def main() -> None:
  """Convert, validate, and write the checkpoint."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--torch-weights", type=Path, required=True)
  parser.add_argument("--model-name", default="esmc_300m", choices=sorted(MODEL_CONFIGS))
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument(
    "--force", action="store_true", help="write even if the acceptance test fails",
  )
  parser.add_argument("--log-level", default="INFO")
  args = parser.parse_args()
  logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

  import torch  # noqa: PLC0415  (late by design: this repo does not depend on torch)

  cfg = MODEL_CONFIGS[args.model_name]
  n_layers = cfg["n_layers"]

  logger.info("loading %s", args.torch_weights)
  state = torch.load(args.torch_weights, map_location="cpu", weights_only=True)
  logger.info("state dict: %d tensors, %.1fM params",
              len(state), sum(v.numel() for v in state.values()) / 1e6)

  skeleton = create_esmc_skeleton(jax.random.PRNGKey(0), args.model_name)
  flat, _ = jax.tree_util.tree_flatten_with_path(eqx.filter(skeleton, eqx.is_array))
  logger.info("skeleton: %d array leaves", len(flat))

  new_leaves = []
  for path, leaf in flat:
    torch_key = path_to_torch_key(jax.tree_util.keystr(path))
    arr = build_leaf(torch_key, leaf.shape, state, n_layers)
    logger.debug("%-52s <- %s", jax.tree_util.keystr(path), torch_key)
    new_leaves.append(arr)

  arrays_only = eqx.filter(skeleton, eqx.is_array)
  loaded_arrays = jax.tree_util.tree_unflatten(
    jax.tree_util.tree_structure(arrays_only), new_leaves,
  )
  model = eqx.combine(loaded_arrays, eqx.filter(skeleton, eqx.is_array, inverse=True))

  recovery, median_std = acceptance_test(model)
  logger.info(
    "ACCEPTANCE  ubiquitin argmax recovery = %.3f  (must exceed %.2f; untrained: 0.000)",
    recovery, MIN_RECOVERY,
  )
  logger.info(
    "ACCEPTANCE  median weight-matrix std  = %.4f  (must be below %.2f; untrained: 1.000)",
    median_std, MAX_MEDIAN_WEIGHT_STD,
  )

  ok = recovery >= MIN_RECOVERY and median_std <= MAX_MEDIAN_WEIGHT_STD
  if not ok and not args.force:
    msg = (
      f"acceptance test FAILED (recovery {recovery:.3f}, median std {median_std:.4f}). "
      f"Refusing to write {args.out}. This is the check whose absence let an untrained "
      f"checkpoint ship. Pass --force only if you know why it fails."
    )
    raise SystemExit(msg)
  if not ok:
    logger.warning("acceptance test failed but --force given; writing anyway")

  args.out.parent.mkdir(parents=True, exist_ok=True)
  hyperparams = {"model_name": args.model_name, "config": cfg}
  with args.out.open("wb") as f:
    f.write((json.dumps(hyperparams) + "\n").encode())
    eqx.tree_serialise_leaves(f, model)
  logger.info("wrote %s (%.2f GB)", args.out, args.out.stat().st_size / 1e9)

  verdict = "PASS" if ok else "FAIL"
  print(f"\nacceptance: recovery={recovery:.3f}  median_std={median_std:.4f}  -> {verdict}")
  print(f"wrote {args.out}")


if __name__ == "__main__":
  main()
