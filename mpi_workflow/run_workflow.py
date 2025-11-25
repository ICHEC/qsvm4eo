import qsvm4eo
import numpy as np
from mpi4py import MPI
from sklearn.svm import SVC

# Load the data
num_features = 4
x_train, y_train, x_test, y_test = qsvm4eo.load_data(
    data_path="..", num_features=num_features, scale_features=False
)

# Encode the data, transforming the features into qubit coordinates
encoding = qsvm4eo.RadialEncoding(
    max_feature=np.max(x_train), shift=1.0, scaling=5.4, n_features=num_features
)
qbits_train = [encoding.encode(x) for x in x_train[:50]]

print("Qubit Geometries:")
print(qbits_train[:4])

# Create a worker for the quantum task
worker = MPI.COMM_SELF.Spawn("python", "./qsvm_train.py", 1)

# Send the qubit geometries
worker.send(qbits_train, dest=0, tag=0)

# Recieve the state probabilities.
probs_train = worker.recv(source=0, tag=1)
print("Probabilities:")
print(probs_train)

# Recieve the end signal
end = worker.recv(source=0, tag=2)
print("Master end signal received, end the job!")

# disconnect from worker
worker.Disconnect()

# Compute the kernel
kernel = qsvm4eo.Kernel()
gram_train = kernel.compute_gram_train(probs_train)

# Fit the SVM
model = SVC(kernel="precomputed")
model.fit(gram_train, y_train)

train_score = model.score(gram_train, y_train[:50])
print("Train acc:", train_score)
