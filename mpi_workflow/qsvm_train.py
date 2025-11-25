import numpy as np
import qsvm4eo
from mpi4py import MPI
from qlmaas.qpus import AnalogQPU
from qat.core import Schedule

comm = MPI.Comm.Get_parent()

qubit_coords = comm.recv(source=0, tag=0)

drive = qsvm4eo.generate_rydberg_hamiltonian(qubit_coords)

duration = 660  # ns
duration /= 1000
schedule = Schedule(drive=drive, tmax=duration)

job = schedule.to_job()

my_qpu = AnalogQPU()
async_result = my_qpu.submit(job)
result = async_result.join()

probs = np.array([r.probability for r in result])
comm.send(probs, dest=0, tag=1)

comm.send(1, dest=0, tag=2)
comm.Disconnect()
