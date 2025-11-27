import qutip
import numpy as np
from .myqlm_hamiltonian import parameters

c6 = parameters["c6"]
amplitude = parameters["amplitude"]


def generate_qutip_hamiltonian(qbits):
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
    def run(self, qbit_coords):
        hamiltonian = generate_qutip_hamiltonian(qbit_coords)
        intial_state = qutip.tensor([qutip.basis(2)] * len(qbit_coords))
        evolved_state = qutip.sesolve(hamiltonian, intial_state, [0.0, 0.66])
        final_state = evolved_state.final_state.full()
        probs = np.real(np.conj(final_state) * final_state)
        return probs.flatten()

    def batch(self, qbits):
        return np.array([self.run(q) for q in qbits])
