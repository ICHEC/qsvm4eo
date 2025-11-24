import pandas as pd
import numpy as np
from features2qubits import RadialEncoding
from mpi4py import MPI

# load data
data_train = pd.read_csv("../data/train_32.csv", header=0)
features = ['B02', 'B03', 'B04', 'B08']

x_train = np.array(data_train[features].values, dtype=float)

encoding = RadialEncoding(
    max_feature=np.max(x_train), shift=1., scaling=5.4, n_features=len(features)
    )
qbits0 = encoding.encode(x_train[0])

print("Qubit Geometry:")
print(qbits0)


worker = MPI.COMM_SELF.Spawn('python', './quantum_task.py', 1)
worker.send(qbits0, dest=0, tag=0)

probs = worker.recv(source=0, tag=1)
print("Probabilities:")
print(probs)

end = worker.recv(source=0, tag=2)
print("Master end signal received, end the job!")

# disconnect from worker
worker.Disconnect()
