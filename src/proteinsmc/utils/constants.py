"""Constants for nucleotide and amino acid sequences, including mappings and codon frequencies."""

from __future__ import annotations

import collections
from venv import logger

import jax.numpy as jnp

restypes = [
  "A",
  "R",
  "N",
  "D",
  "C",
  "Q",
  "E",
  "G",
  "H",
  "I",
  "L",
  "K",
  "M",
  "F",
  "P",
  "S",
  "T",
  "W",
  "Y",
  "V",
]
restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)  # := 20.
unk_restype_index = restype_num  # Catch-all index for unknown restypes.

restypes_with_x = [*restypes, "X"]
restype_order_with_x = {restype: i for i, restype in enumerate(restypes_with_x)}
order_aa = {v: k for k, v in restype_order.items()}

NUCLEOTIDES_CHAR = ["A", "C", "G", "T"]
NUCLEOTIDES_INT_MAP = {n: i for i, n in enumerate(NUCLEOTIDES_CHAR)}
INT_TO_NUCLEOTIDES_CHAR_MAP = dict(enumerate(NUCLEOTIDES_CHAR))
NUCLEOTIDES_JAX = jnp.array([NUCLEOTIDES_INT_MAP[n] for n in NUCLEOTIDES_CHAR])
NUCLEOTIDES_NUM_STATES = len(NUCLEOTIDES_CHAR)

CODON_TO_RES_CHAR = {
  "TCA": "S",
  "TCC": "S",
  "TCG": "S",
  "TCT": "S",
  "TTC": "F",
  "TTT": "F",
  "TTA": "L",
  "TTG": "L",
  "TAC": "Y",
  "TAT": "Y",
  "TAA": "X",
  "TAG": "X",
  "TGC": "C",
  "TGT": "C",
  "TGA": "X",
  "TGG": "W",
  "CTA": "L",
  "CTC": "L",
  "CTG": "L",
  "CTT": "L",
  "CCA": "P",
  "CCC": "P",
  "CCG": "P",
  "CCT": "P",
  "CAC": "H",
  "CAT": "H",
  "CAA": "Q",
  "CAG": "Q",
  "CGA": "R",
  "CGC": "R",
  "CGG": "R",
  "CGT": "R",
  "ATA": "I",
  "ATC": "I",
  "ATT": "I",
  "ATG": "M",
  "ACA": "T",
  "ACC": "T",
  "ACG": "T",
  "ACT": "T",
  "AAC": "N",
  "AAT": "N",
  "AAA": "K",
  "AAG": "K",
  "AGC": "S",
  "AGT": "S",
  "AGA": "R",
  "AGG": "R",
  "GTA": "V",
  "GTC": "V",
  "GTG": "V",
  "GTT": "V",
  "GCA": "A",
  "GCC": "A",
  "GCG": "A",
  "GCT": "A",
  "GAC": "D",
  "GAT": "D",
  "GAA": "E",
  "GAG": "E",
  "GGA": "G",
  "GGC": "G",
  "GGG": "G",
  "GGT": "G",
}

AA_CHAR_TO_INT_MAP = restype_order
INT_TO_AA_CHAR_MAP = dict(enumerate(restypes))
PROTEINMPNN_X_INT = 21
STOP_INT = PROTEINMPNN_X_INT
UNKNOWN_AA_INT = PROTEINMPNN_X_INT
MAX_NUC_INT = len(NUCLEOTIDES_CHAR) - 1
AMINO_ACIDS_NUM_STATES = 20

CODON_INT_TO_RES_INT_JAX = jnp.full(
  (MAX_NUC_INT + 1, MAX_NUC_INT + 1, MAX_NUC_INT + 1),
  UNKNOWN_AA_INT,
  dtype=jnp.int8,
)
for codon_str, res_char in CODON_TO_RES_CHAR.items():
  n1, n2, n3 = (NUCLEOTIDES_INT_MAP[c] for c in codon_str)
  if res_char in AA_CHAR_TO_INT_MAP:
    aa_int = AA_CHAR_TO_INT_MAP[res_char]
    CODON_INT_TO_RES_INT_JAX = CODON_INT_TO_RES_INT_JAX.at[n1, n2, n3].set(aa_int)

ECOLI_CODON_FREQ_CHAR = {
  "TTT": 19.7,
  "TTC": 15.0,
  "TTA": 15.2,
  "TTG": 11.9,
  "CTT": 11.9,
  "CTC": 10.5,
  "CTA": 5.3,
  "CTG": 46.9,
  "ATT": 30.5,
  "ATC": 18.2,
  "ATA": 3.7,
  "ATG": 24.8,
  "GTT": 16.8,
  "GTC": 11.7,
  "GTA": 11.5,
  "GTG": 26.4,
  "TCT": 5.7,
  "TCC": 5.5,
  "TCA": 7.8,
  "TCG": 8.0,
  "AGT": 7.2,
  "AGC": 16.6,
  "CCT": 8.4,
  "CCC": 6.4,
  "CCA": 6.6,
  "CCG": 26.7,
  "ACT": 8.0,
  "ACC": 22.8,
  "ACA": 6.4,
  "ACG": 11.5,
  "GCT": 10.7,
  "GCC": 31.6,
  "GCA": 21.1,
  "GCG": 38.5,
  "TAT": 16.8,
  "TAC": 14.6,
  "CAT": 15.8,
  "CAC": 13.1,
  "CAA": 12.1,
  "CAG": 27.7,
  "AAT": 21.9,
  "AAC": 24.4,
  "AAA": 33.2,
  "AAG": 12.1,
  "GAT": 37.9,
  "GAC": 20.5,
  "GAA": 43.7,
  "GAG": 18.4,
  "TGT": 5.9,
  "TGC": 8.0,
  "TGG": 10.7,
  "CGT": 21.1,
  "CGC": 26.0,
  "CGA": 4.3,
  "CGG": 4.1,
  "AGA": 1.4,
  "AGG": 1.6,
  "TAA": 1.8,
  "TAG": 0.0,
  "TGA": 1.0,
  "GGA": 9.5,
  "GGC": 27.1,
  "GGG": 20.5,
  "GGT": 11.3,
}
ECOLI_CODON_FREQ_JAX = jnp.zeros(
  (MAX_NUC_INT + 1, MAX_NUC_INT + 1, MAX_NUC_INT + 1),
  dtype=jnp.float32,
)
for codon_str, freq in ECOLI_CODON_FREQ_CHAR.items():
  n1, n2, n3 = (NUCLEOTIDES_INT_MAP[c] for c in codon_str)
  ECOLI_CODON_FREQ_JAX = ECOLI_CODON_FREQ_JAX.at[n1, n2, n3].set(freq)

RES_TO_CODON_CHAR = collections.defaultdict(list)
for codon_str, res_char in CODON_TO_RES_CHAR.items():
  if res_char != "X":
    RES_TO_CODON_CHAR[res_char].append(codon_str)

ECOLI_MAX_FREQS_JAX_list = [0.0] * (len(AA_CHAR_TO_INT_MAP) + 1)
for aa_char, aa_int_colabdesign in restype_order.items():
  if aa_char == "X":
    ECOLI_MAX_FREQS_JAX_list[aa_int_colabdesign] = 1.0
    continue
  max_f = 0.0
  for codon_str in RES_TO_CODON_CHAR.get(aa_char, []):
    max_f = max(max_f, ECOLI_CODON_FREQ_CHAR.get(codon_str, 0.0))
  ECOLI_MAX_FREQS_JAX_list[aa_int_colabdesign] = max(max_f, 1e-9)
ECOLI_MAX_FREQS_JAX = jnp.array(ECOLI_MAX_FREQS_JAX_list, dtype=jnp.float32)


ESM_SEQUENCE_VOCAB = SEQUENCE_VOCAB = [
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
CHAIN_BREAK_STR = "|"

SEQUENCE_BOS_STR = "<cls>"
SEQUENCE_EOS_STR = "<eos>"

MASK_STR_SHORT = "_"
SEQUENCE_MASK_STR = "<mask>"
ESM_AA_CHAR_TO_INT_MAP = {c: i for i, c in enumerate(ESM_SEQUENCE_VOCAB)}

ESM_BOS_ID = ESM_AA_CHAR_TO_INT_MAP["<cls>"]
ESM_PAD_ID = ESM_AA_CHAR_TO_INT_MAP["<pad>"]
ESM_EOS_ID = ESM_AA_CHAR_TO_INT_MAP["<eos>"]
ESM_UNK_ID = ESM_AA_CHAR_TO_INT_MAP["<unk>"]
ESM_MASK_ID = ESM_AA_CHAR_TO_INT_MAP["<mask>"]
# --- Amino-acid integer -> ESM token maps ------------------------------------------
#
# TWO distinct source alphabets reach ESM through this module, and conflating them
# silently permutes every residue that is not a fixed point of the permutation. They are
# therefore built and named separately. See
# `.praxia/docs/research/260813_alphabet-provenance-trace.md`.
#
#   AlphaFold  — this package's own encoding (`restypes`, above):  ARNDCQEGHILKMFPSTWYV
#                Produced by `string_to_int_sequence` and fed to `scoring/esm.py`.
#   ProteinMPNN — asr's declared canonical (`asr/src/asr/alphabet.py:8`):
#                                                                  ACDEFGHIKLMNPQRSTVWY
#                Supplied by external callers of `utils.esm.remap_sequences`, whose
#                docstring has always promised the ProteinMPNN scheme.
#
# Before 2026-08-13 a single table was built from the AlphaFold ordering while carrying
# the ProteinMPNN name, so external callers honouring the documented contract had their
# sequences permuted. Only A, S and T are fixed points of that permutation.
#
# Both tables are sized to cover every legal sequence value — including the gap/X index
# 20 used by asr and the stop/unknown sentinel `PROTEINMPNN_X_INT` (21). The previous
# table was length 20, so a JAX gather clamped indices 20 and 21 to Valine.

PROTEINMPNN_RESTYPES = "ACDEFGHIKLMNPQRSTVWY"
"""ProteinMPNN's 20-letter ordering. Distinct from `restypes`, which is AlphaFold's."""

_ESM_MAP_LEN = PROTEINMPNN_X_INT + 1
"""Covers indices 0..21 inclusive, so no legal sequence value can clamp."""


def _build_esm_token_map(source_alphabet: str, alphabet_name: str) -> jnp.ndarray:
  """Build a source-alphabet-index -> ESM-token lookup table.

  Every index outside `source_alphabet` — notably the gap/X index 20 and the stop
  sentinel 21 — resolves to `<unk>` explicitly rather than by out-of-bounds clamping.
  """
  table = jnp.full(_ESM_MAP_LEN, ESM_UNK_ID, dtype=jnp.int32)
  for idx, char in enumerate(source_alphabet):
    if char in ESM_AA_CHAR_TO_INT_MAP:
      table = table.at[idx].set(ESM_AA_CHAR_TO_INT_MAP[char])
    else:
      msg = (
        f"{alphabet_name} character '{char}' (int {idx}) not found in "
        f"ESM vocabulary. Mapping to UNK."
      )
      logger.warning(msg)
  return table


PROTEINMPNN_TO_ESM_AA_MAP_JAX = _build_esm_token_map(PROTEINMPNN_RESTYPES, "ProteinMPNN")
"""ProteinMPNN-ordered integer -> ESM token. Matches this constant's name and the
documented contract of `utils.esm.remap_sequences`."""

ALPHAFOLD_TO_ESM_AA_MAP_JAX = _build_esm_token_map("".join(restypes), "AlphaFold")
"""AlphaFold-ordered integer -> ESM token. This is what `scoring/esm.py` needs, because
this package's own encoders emit AlphaFold-ordered integers."""
