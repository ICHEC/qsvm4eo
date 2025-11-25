import qsvm4eo
import numpy as np
from mpi4py import MPI

# Load the data
num_features = 4
x_train, y_train, x_test, y_test = qsvm4eo.load_data(
    data_path="..", num_features=num_features, scale_features=False
)

# Encode the data, transforming the features into qubit coordinates
encoding = qsvm4eo.RadialEncoding(
    max_feature=np.max(x_train), shift=1.0, scaling=5.4, n_features=num_features
)
qbits0 = encoding.encode(x_train[0])

print("Qubit Geometry:")
print(qbits0)

# Create a worker for the quantum task
worker = MPI.COMM_SELF.Spawn("python", "./qsvm_train.py", 1)

# Send the qubit geometries
worker.send(qbits0, dest=0, tag=0)

# Recieve the state probabilities.
probs = worker.recv(source=0, tag=1)
print("Probabilities:")
print(probs)

# Recieve the end signal
end = worker.recv(source=0, tag=2)
print("Master end signal received, end the job!")

# disconnect from worker
worker.Disconnect()
