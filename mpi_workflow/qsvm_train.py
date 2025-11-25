import numpy as np
import qsvm4eo
from mpi4py import MPI
from qlmaas.qpus import AnalogQPU
from qat.core import Schedule

# Get the parent node
comm = MPI.Comm.Get_parent()

# Receive the qubit geometries
qubit_coords = comm.recv(source=0, tag=0)

# Create jobs
schedule = Schedule(
    drive=qsvm4eo.generate_rydberg_hamiltonian(qubit_coords), tmax=0.66  # μs
)
job = schedule.to_job()

# Run jobs
my_qpu = AnalogQPU()
async_result = my_qpu.submit(job)
result = async_result.join()

# Get state probabilities
probs = np.array([r.probability for r in result])

# Send state probabilities
comm.send(probs, dest=0, tag=1)

# Send end signal
comm.send(1, dest=0, tag=2)
comm.Disconnect()
