import numpy as np
import qsvm4eo
from mpi4py import MPI
from qlmaas.qpus import AnalogQPU
from qat.core import Batch, Schedule

# Get the parent node
comm = MPI.Comm.Get_parent()

# Receive the qubit geometries
qbits_train = comm.recv(source=0, tag=0)

# Define the QPU
my_qpu = AnalogQPU()

# Create jobs
duration = 0.66  # μs
schedules = [
    Schedule(drive=qsvm4eo.generate_myqlm_hamiltonian(qbits), tmax=duration)
    for qbits in qbits_train
]
jobs = [schedule.to_job() for schedule in schedules]

# Run jobs
async_result = my_qpu.submit(Batch(jobs))
results = async_result.join()

# Get state probabilities
probs = np.array([[r.probability for r in result] for result in results])

# Send state probabilities
comm.send(probs, dest=0, tag=1)

# Send end signal
comm.send(1, dest=0, tag=2)
comm.Disconnect()
