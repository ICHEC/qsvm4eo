import qutip
import numpy as np
from .myqlm_hamiltonian import parameters

c6 = parameters["c6"]
amplitude = parameters["amplitude"]

def _generate_qutip_hamiltonian(qbits):
    """Generate Rydberg Hamiltonian."""
    hamiltonian = 0.0

    nqbits = len(qbits)

    for i in range(nqbits):
        op = [qutip.qeye(2)] * nqbits
        op[i] = qutip.sigmax()
        hamiltonian += amplitude * 0.5 * qutip.tensor(op)

    for i in range(nqbits - 1):
        for j in range(i + 1, nqbits):
            op = [qutip.qeye(2)] * nqbits
            op[i] = qutip.num(2)
            op[j] = qutip.num(2)

            rij = np.linalg.norm(qbits[i] - qbits[j])
            hamiltonian += (c6 / rij**6) * qutip.tensor(op)

    return hamiltonian


class QutipBackend:
    """
    A QPU with a statevector emulator in Qutip.

    Parameters
    ----------
    duration : float
        The time in μs to evolve the system. Defaults to 0.66.
    """
    def __init__(self, duration=0.66):
        self.duration = duration

    def run(self, qbit_coords):
        """
        Run a simulator for a single qubit configuration.

        Parameters
        ----------
        qbit_coords : np.ndarray
            An np.ndarray of shape (N, 2) where N is the number of qubits,
            specifying the qubit coordinates.

        Returns
        -------
        np.ndarray
            An array of length 2^N containing the state probabilities.
        """
        hamiltonian = _generate_qutip_hamiltonian(qbit_coords)
        intial_state = qutip.tensor([qutip.basis(2)] * len(qbit_coords))
        evolved_state = qutip.sesolve(hamiltonian, intial_state, [0.0, self.duration])
        final_state = evolved_state.final_state.full()
        probs = np.real(np.conj(final_state) * final_state)
        return probs.flatten()

    def batch(self, qbits):
        """
        Run a simulator for a batch of qubit configurations.

        Parameters
        ----------
        qbits : np.ndarray
            An np.ndarray of shape (B, N, 2) where B is the number of
            qubit configurations and N is the number of qubits, which
            specifies the qubit coordinates.

        Returns
        -------
        np.ndarray
            An array of shape (B,2^N) containing the state probabilities.
        """
        return np.array([self.run(q) for q in qbits])
