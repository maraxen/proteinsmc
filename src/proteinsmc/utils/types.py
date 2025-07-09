"""Type aliases for nucleotide and protein sequences."""

from colabdesign.mpnn.model import mk_mpnn_model  # type: ignore[import]
from jaxtyping import Array, Bool, Float, Int

NucleotideSequence = Int[Array, "nucleotide_sequence_length"]
ProteinSequence = Int[Array, "protein_sequence_length"]
EvoSequence = NucleotideSequence | ProteinSequence
PopulationNucleotideSequences = Int[Array, "population_size nucleotide_sequence_length"]
PopulationProteinSequences = Int[Array, "population_size protein_sequence_length"]
PopulationSequences = PopulationNucleotideSequences | PopulationProteinSequences
PopulationSequenceFloats = Float[Array, "population_size"]
PopulationSequenceBools = Bool[Array, "population_size"]
ScalarFloat = Float[Array, ""]
ScalarBool = Bool[Array, ""]
ScalarInt = Int[Array, ""]
StackedPopulationSequenceFloats = Float[PopulationSequenceFloats, "population_size combine_funcs"]
FunctionFloats = Float[Array, "n_fitness_funcs"]
MPNNModel = mk_mpnn_model
FitnessWeights = Float[Array, "n_fitness_funcs"]
IslandFloats = Float[Array, "n_islands"]
PerGenerationFloat = Float[Array, "generations"]
