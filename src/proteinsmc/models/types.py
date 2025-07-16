"""Type aliases for nucleotide and protein sequences."""

from colabdesign.mpnn.model import mk_mpnn_model  # type: ignore[import]
from jaxtyping import Array, Int

NucleotideSequence = Int[Array, "nucleotide_sequence_length n_seqs"]
ProteinSequence = Int[Array, "protein_sequence_length n_seqs"]
EvoSequence = NucleotideSequence | ProteinSequence
MPNNModel = mk_mpnn_model
