# Project Roadmap

This document outlines the future development plan for the Protein SMC Experiment project.

## Near-Term Goals

- **Implement Protein-to-Nucleotide Conversion:** The `sequence_agnostic` decorator currently only supports nucleotide-to-protein conversion. The next step is to implement the reverse conversion to make the decorator fully bidirectional.
- **Expand Fitness Functions:** Add more fitness functions to the library, such as those related to protein stability, solubility, and binding affinity. This will allow for more complex and realistic protein design experiments.
- **Improve Benchmarking:** Develop a comprehensive benchmarking suite to compare the performance of the parallel replica SMC algorithm against other sampling methods. This should include a variety of test cases and metrics.

## Mid-Term Goals

- **Enhance Evolutionary Studies:** Extend the framework to support more advanced evolutionary studies, such as ancestral sequence reconstruction and the simulation of different evolutionary pressures.
- **Improve Visualization:** Develop more sophisticated visualization tools to analyze the results of the simulations. This could include interactive plots of fitness landscapes, sequence diversity, and other metrics.
- **Integrate with External Tools:** Add support for integrating the framework with other protein design and analysis tools, such as Rosetta and AlphaFold.

## Long-Term Goals

- **Develop a User-Friendly Interface:** Create a user-friendly interface for setting up and running experiments. This could be a web-based interface or a command-line tool with a more intuitive syntax.
- **Expand to Other Biomolecules:** Extend the framework to support the design and analysis of other biomolecules, such as RNA and DNA.
- **Open-Source Community:** Foster an open-source community around the project to encourage collaboration and accelerate development.
