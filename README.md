# qsvm4eo
`qsvm4eo` is a package for running Support Vector Machines (SVMs) computed with a quantum kernel for Earth Observation data.

The quantum kernel is based on a analogue computing framework as introduced by Henry et al. (for details see [here](https://arxiv.org/pdf/2107.03247) and [here](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.107.042615)). While this scheme was initially intended for objects with a graph topology, we have developed a number of encoding schemes to allow for generic feature vectors to be represented.

## Installation
Clone the repo and (making sure you’re in the directory where the `pyproject.toml` file is situated) install the package and its dependencies using `pip` (to install in editbale mode use the `-e` flag)
```
pip install .
```

## Background and Theory

### Earth Observation Data

### Analogue Quantum Kernel
The quantum kernel computed here is designed for analgue quantum computers, a common choice of modality for implementing this is neutral atom quantum computers. The main idea here (as developed by Henry et al.) is to encode the feature vector data into the positions and topology of the qubits. As we are dealing with data which is not in a natural graph format we must introduce an encoding scheme as discussed below. A constant pulse is applied to the system and it is left to evolve up to some specfied time. The states of the qubits are then measured and from this a probability distribution can be constructed (for example the distribution of the total number of excitations). Using a probability similarity measure, for example the Jensen–Shannon divergence, the similarity between the distributions can be computed and a suitable kernel for a SVM can be created.

### Encoding Strategies