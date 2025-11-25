import numpy as np
from qat.core import Observable, Term


c6 = 865723.02  # (rad/µs)(µm)**6
duration = 660  # ns
amplitude = 2 * np.pi  # rad/µs


def generate_myqlm_hamiltonian(qbits):
    """Generate Rydberg Hamiltonian."""
    nqbits = len(qbits)

    amplitude_term = Observable(
        nqbits, pauli_terms=[Term(0.5, "X", [i]) for i in range(nqbits)]
    )

    interaction_term = 0
    for i in range(nqbits - 1):
        for j in range(i + 1, nqbits):
            rij = np.linalg.norm(qbits[i] - qbits[j])
            interaction_term += (1 / rij**6) * _occ_op(nqbits, i) * _occ_op(nqbits, j)

    return [
        (amplitude, amplitude_term),
        (c6, interaction_term),
    ]


def _occ_op(nqbits, qi):
    return (1 - Observable(nqbits, pauli_terms=[Term(1.0, "Z", [qi])])) / 2
