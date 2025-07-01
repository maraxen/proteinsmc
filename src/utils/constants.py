import collections

import jax.numpy as jnp
from colabdesign.mpnn.model import (  # type: ignore[import]
    aa_order as colabdesign_aa_order,
)
from colabdesign.mpnn.model import (
    order_aa as colabdesign_order_aa,
)

# --- Constants and Mappings ---
# Nucleotide mappings remain the same
NUCLEOTIDES_CHAR = ['A', 'C', 'G', 'T']
NUCLEOTIDES_INT_MAP = {n: i for i, n in enumerate(NUCLEOTIDES_CHAR)}
INT_TO_NUCLEOTIDES_CHAR_MAP = {i: n for i, n in enumerate(NUCLEOTIDES_CHAR)}
NUCLEOTIDES_JAX = jnp.array([NUCLEOTIDES_INT_MAP[n] for n in NUCLEOTIDES_CHAR])

# Codon to Amino Acid Character mapping (unchanged)
CODON_TO_RES_CHAR = {
    "TCA": 'S', "TCC": 'S', "TCG": 'S', "TCT": 'S', "TTC": 'F', "TTT": 'F',
    "TTA": 'L', "TTG": 'L', "TAC": 'Y', "TAT": 'Y', "TAA": 'X', "TAG": 'X',
    "TGC": 'C', "TGT": 'C', "TGA": 'X', "TGG": 'W', "CTA": 'L', "CTC": 'L',
    "CTG": 'L', "CTT": 'L', "CCA": 'P', "CCC": 'P', "CCG": 'P', "CCT": 'P',
    "CAC": 'H', "CAT": 'H', "CAA": 'Q', "CAG": 'Q', "CGA": 'R', "CGC": 'R',
    "CGG": 'R', "CGT": 'R', "ATA": 'I', "ATC": 'I', "ATT": 'I', "ATG": 'M',
    "ACA": 'T', "ACC": 'T', "ACG": 'T', "ACT": 'T', "AAC": 'N', "AAT": 'N',
    "AAA": 'K', "AAG": 'K', "AGC": 'S', "AGT": 'S', "AGA": 'R', "AGG": 'R',
    "GTA": 'V', "GTC": 'V', "GTG": 'V', "GTT": 'V', "GCA": 'A', "GCC": 'A',
    "GCG": 'A', "GCT": 'A', "GAC": 'D', "GAT": 'D', "GAA": 'E', "GAG": 'E',
    "GGA": 'G', "GGC": 'G', "GGG": 'G', "GGT": 'G',
}

# Amino Acid Mappings are now directly from colabdesign
AA_CHAR_TO_INT_MAP = colabdesign_aa_order
INT_TO_AA_CHAR_MAP = {i: char for i, char in enumerate(colabdesign_order_aa)}
COLABDESIGN_X_INT = 21
STOP_INT = COLABDESIGN_X_INT
UNKNOWN_AA_INT = COLABDESIGN_X_INT
MAX_NUC_INT = len(NUCLEOTIDES_CHAR) - 1

# Initialize codon to (colabdesign) integer AA JAX table
CODON_INT_TO_RES_INT_JAX = jnp.full(
    (MAX_NUC_INT + 1, MAX_NUC_INT + 1, MAX_NUC_INT + 1),
    UNKNOWN_AA_INT, # Default to 'X'
    dtype=jnp.int32
)
for codon_str, res_char in CODON_TO_RES_CHAR.items():
    n1, n2, n3 = (NUCLEOTIDES_INT_MAP[c] for c in codon_str)
    if res_char in AA_CHAR_TO_INT_MAP:
        aa_int = AA_CHAR_TO_INT_MAP[res_char]
        CODON_INT_TO_RES_INT_JAX = CODON_INT_TO_RES_INT_JAX.at[n1, n2, n3].set(aa_int)

# E. coli codon frequencies (character-based, unchanged)
ECOLI_CODON_FREQ_CHAR = {
    "TTT": 19.7, "TTC": 15.0, "TTA": 15.2, "TTG": 11.9, "CTT": 11.9, 
    "CTC": 10.5, "CTA": 5.3, "CTG": 46.9, "ATT": 30.5, "ATC": 18.2, 
    "ATA": 3.7, "ATG": 24.8, "GTT": 16.8, "GTC": 11.7, "GTA": 11.5, 
    "GTG": 26.4, "TCT": 5.7, "TCC": 5.5, "TCA": 7.8, "TCG": 8.0, 
    "AGT": 7.2, "AGC": 16.6, "CCT": 8.4, "CCC": 6.4, "CCA": 6.6, 
    "CCG": 26.7, "ACT": 8.0, "ACC": 22.8, "ACA": 6.4, "ACG": 11.5, 
    "GCT": 10.7, "GCC": 31.6, "GCA": 21.1, "GCG": 38.5, "TAT": 16.8, 
    "TAC": 14.6, "CAT": 15.8, "CAC": 13.1, "CAA": 12.1, "CAG": 27.7, 
    "AAT": 21.9, "AAC": 24.4, "AAA": 33.2, "AAG": 12.1, "GAT": 37.9, 
    "GAC": 20.5, "GAA": 43.7, "GAG": 18.4, "TGT": 5.9, "TGC": 8.0, 
    "TGG": 10.7, "CGT": 21.1, "CGC": 26.0, "CGA": 4.3, "CGG": 4.1, 
    "AGA": 1.4, "AGG": 1.6, "TAA": 1.8, "TAG": 0.0, "TGA": 1.0
}
ECOLI_CODON_FREQ_JAX = jnp.zeros(
    (MAX_NUC_INT + 1, MAX_NUC_INT + 1, MAX_NUC_INT + 1), dtype=jnp.float32
)
for codon_str, freq in ECOLI_CODON_FREQ_CHAR.items():
    n1, n2, n3 = (NUCLEOTIDES_INT_MAP[c] for c in codon_str)
    ECOLI_CODON_FREQ_JAX = ECOLI_CODON_FREQ_JAX.at[n1, n2, n3].set(freq)

# Compute ECOLI_MAX_FREQS_JAX
_res_to_codons_char = collections.defaultdict(list)
for codon, res in CODON_TO_RES_CHAR.items():
    if res != 'X':
        _res_to_codons_char[res].append(codon)
        
RES_TO_CODON_CHAR = {
    'M': "ATG", 'T': "ACC", 'P': "CCG", 'K': "AAG", 'F': "TTT", 'L': "CTG",
    'I': "ATT", 'V': "GTG", 'C': "TGC", 'S': "AGC", 'A': "GCC", 'R': "CGC",
    'N': "AAC", 'D': "GAT", 'G': "GGC", 'Q': "CAG", 'Y': "TAT", 'W': "TGG",
    'E': "GAA", 'H': "CAC", 'X': "TAA"
}

num_colabdesign_aas = len(colabdesign_order_aa)
ECOLI_MAX_FREQS_JAX_list = [0.0] * num_colabdesign_aas
for aa_char, aa_int_colabdesign in colabdesign_aa_order.items():
    if aa_char == 'X':
        ECOLI_MAX_FREQS_JAX_list[aa_int_colabdesign] = 1.0
        continue
    max_f = 0.0
    for codon_str in _res_to_codons_char.get(aa_char, []):
        max_f = max(max_f, ECOLI_CODON_FREQ_CHAR.get(codon_str, 0.0))
    ECOLI_MAX_FREQS_JAX_list[aa_int_colabdesign] = max(max_f, 1e-9)
ECOLI_MAX_FREQS_JAX = jnp.array(ECOLI_MAX_FREQS_JAX_list, dtype=jnp.float32)
