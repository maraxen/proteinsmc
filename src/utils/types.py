from colabdesign.mpnn.model import mk_mpnn_model  # type: ignore[import]
from jaxtyping import Array, Bool, Float, Int

# Type aliases for common shape descriptors
# These tell the linter that these are intentional names for shape annotations
NucleotideSequence = Int[Array, "nucleotide_sequence_length"]
ProteinSequence = Int[Array, "protein_sequence_length"]
EvoSequence = NucleotideSequence | ProteinSequence
PopulationNucleotideSequences = Int[Array, "n_particles nucleotide_sequence_length"]
PopulationProteinSequences = Int[Array, "n_particles protein_sequence_length"]
PopulationSequences = PopulationNucleotideSequences | PopulationProteinSequences
PopulationSequenceFloats = Float[Array, "n_particles"]
PopulationSequenceBools = Bool[Array, "n_particles"]
ScalarFloat = Float[Array, ""]
ScalarBool = Bool[Array, ""]
ScalarInt = Int[Array, ""]
MPNNModel = mk_mpnn_model
FitnessWeights = Float[Array, "n_fitness_funcs"]
