from colabdesign.mpnn.model import mk_mpnn_model  # type: ignore[import]
from jaxtyping import Array, Bool, Float, Int

# Type aliases for common shape descriptors
# These tell the linter that these are intentional names for shape annotations
NucleotideSequence = Int[Array, "nuc_len"]
ProteinSequence = Int[Array, "protein_len"]
MCSequence = NucleotideSequence | ProteinSequence
BatchNucleotideSequences = Int[Array, "n_particles nuc_len"]
BatchProteinSequences = Int[Array, "n_particles protein_len"]
BatchSequenceFloats = Float[Array, "n_particles"]
BatchSequenceBools = Bool[Array, "n_particles"]
ScalarFloat = Float[Array, ""]
ScalarBool = Bool[Array, ""]
ScalarInt = Int[Array, ""]
MPNNModel = mk_mpnn_model
FitnessWeights = Float[Array, "n_fitness_funcs"]
BatchSequences = BatchNucleotideSequences | BatchProteinSequences
