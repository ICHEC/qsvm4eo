import qsvm4eo
import numpy as np
from mpi4py import MPI

# load data
num_features = 4
x_train, y_train, x_test, y_test = qsvm4eo.load_data(
    data_path="..", num_features=num_features, scale_features=False
    )

# encode the data
encoding = qsvm4eo.RadialEncoding(
    max_feature=np.max(x_train), shift=1., scaling=5.4, n_features=len(num_features)
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
