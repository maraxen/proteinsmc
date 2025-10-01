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
PROTEINMPNN_TO_ESM_AA_MAP_JAX = jnp.full(
  AMINO_ACIDS_NUM_STATES,  # Size is based on ColabDesign's max AA int
  ESM_UNK_ID,
  dtype=jnp.int32,
)

# Populate the mapping
for cd_char, cd_int in AA_CHAR_TO_INT_MAP.items():
  if cd_char in ESM_AA_CHAR_TO_INT_MAP:
    esm_int = ESM_AA_CHAR_TO_INT_MAP[cd_char]
    PROTEINMPNN_TO_ESM_AA_MAP_JAX = PROTEINMPNN_TO_ESM_AA_MAP_JAX.at[cd_int].set(esm_int)
  else:
    msg = (
      f"ColabDesign character '{cd_char}' (int {cd_int}) not found in "
      f"ESM vocabulary. Mapping to UNK."
    )
    logger.warning(msg)

if "X" in AA_CHAR_TO_INT_MAP and "X" in ESM_AA_CHAR_TO_INT_MAP:
  PROTEINMPNN_TO_ESM_AA_MAP_JAX = PROTEINMPNN_TO_ESM_AA_MAP_JAX.at[AA_CHAR_TO_INT_MAP["X"]].set(
    ESM_AA_CHAR_TO_INT_MAP["X"],
  )
elif "X" in AA_CHAR_TO_INT_MAP:
  # If ESM doesn't have 'X' as a regular token, map to UNK
  PROTEINMPNN_TO_ESM_AA_MAP_JAX = PROTEINMPNN_TO_ESM_AA_MAP_JAX.at[AA_CHAR_TO_INT_MAP["X"]].set(
    ESM_UNK_ID,
  )
