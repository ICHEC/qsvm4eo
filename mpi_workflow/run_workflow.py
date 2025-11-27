import qsvm4eo
import numpy as np
import json
import sys
import datetime
from mpi4py import MPI
from sklearn.svm import SVC

_, job_id = sys.argv

def compute_excitations(qbits):
    # Create a worker for the quantum task
    worker = MPI.COMM_SELF.Spawn("python", "./quantum_task.py", 1)

    # Send the qubit geometries
    print("Computing probabilities")
    worker.send(qbits, dest=0, tag=0)

    # Recieve the state probabilities.
    probs = worker.recv(source=0, tag=1)
    print("Finished computing probabilities:")
    print(probs.shape)

    # Recieve the end signal
    end = worker.recv(source=0, tag=2)
    print("Master end signal received, end the job!")

    # disconnect from worker
    worker.Disconnect()
    return qsvm4eo.compute_excitation_count(probs)


# Load the data
num_features = 4
x_train, y_train, x_test, y_test = qsvm4eo.load_data(
    data_path="..", num_features=num_features, scale_features=False
)

# Encode the data, transforming the features into qubit coordinates
encoding = qsvm4eo.RadialEncoding(
    max_feature=np.max(x_train), shift=1.0, scaling=5.4, n_features=num_features
)
qbits_train = [encoding.encode(x) for x in x_train]
qbits_test = [encoding.encode(x) for x in x_test]

print("Qubit Geometries:")
print(qbits_train[:4])

print("Training:")
excitations_train = compute_excitations(qbits_train)

# Fit the SVM and get the score
kernel = qsvm4eo.Kernel()
gram_train = kernel.compute_gram_train(excitations_train)  # Compute the kernel
model = SVC(kernel="precomputed")
model.fit(gram_train, y_train)
train_score = model.score(gram_train, y_train)

print("Testing")
excitations_test = compute_excitations(qbits_test)

# Compute the kernel and score
gram_test = kernel.compute_gram_test(excitations_test)
y_test_pred = model.predict(gram_test)
test_score = model.score(gram_test, y_test)

print("Train acc:", train_score)
print("Test acc:", test_score)

results = {
    "train_acc": train_score,
    "test_acc": test_score,
    "labels_pred": y_test_pred.tolist(),
    "labels_true": y_test.tolist(),
    "time": datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"),
}

with open("results.json", "w") as fp:
    json.dump(results, fp)
