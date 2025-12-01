import qsvm4eo
import numpy as np
import json
import datetime
from mpi4py import MPI
from sklearn.svm import SVC

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output_name", type=str, default="results")
parser.add_argument("-nfeat", "--num_features", type=int, choices=[4, 8], default=4)
parser.add_argument("-reg", "--regularization", type=float, default=1.0)
args = parser.parse_args()


def compute_distributions(qbits, excitations=True):
    """
    Compute the quantum distributions from the qubit geometries.
    """
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
    if excitations:
        print("Computing the excitations")
        return qsvm4eo.compute_excitation_count(probs)
    else:
        return probs


# Load the data
print("Loading the data")
x_train, y_train, x_test, y_test, label_names = qsvm4eo.load_data(
    data_path="..", num_features=args.num_features, scale_features=False
)

print("Encoding the data into qubits")
# Encode the data, transforming the features into qubit coordinates
encoding = qsvm4eo.RadialEncoding(
    max_feature=np.max(x_train), shift=1.0, scaling=5.4, n_features=args.num_features
)
qbits_train = [encoding.encode(x) for x in x_train]
qbits_test = [encoding.encode(x) for x in x_test]

print("Qubit Geometries:")
print(qbits_train[:4])

print("Computing state probabilities for training set")
probs_train = compute_distributions(qbits_train)

# Fit the SVM and get the score
print("Fitting the SVM")
kernel = qsvm4eo.Kernel()
gram_train = kernel.compute_gram_train(probs_train)  # Compute the kernel
model = SVC(kernel="precomputed", C=args.regularization)
model.fit(gram_train, y_train)
train_score = model.score(gram_train, y_train)

print("Computing state probabilities for test set")
probs_test = compute_distributions(qbits_test)

# Compute the kernel and score
print("Testing the SVM")
gram_test = kernel.compute_gram_test(probs_test, probs_train)
y_test_pred = model.predict(gram_test)
test_score = model.score(gram_test, y_test)

print("Train acc:", train_score)
print("Test acc:", test_score)

results = {
    "number_of_features": args.num_features,
    "regularization": args.regularization,
    "train_acc": train_score,
    "test_acc": test_score,
    "labels_pred": y_test_pred.tolist(),
    "labels_true": y_test.tolist(),
    "label_names": label_names,
    "time": datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"),
}

with open(f"results_{args.output_name}.json", "w") as fp:
    json.dump(results, fp)
