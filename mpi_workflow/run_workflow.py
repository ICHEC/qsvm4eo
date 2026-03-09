import argparse
import datetime
import json

import numpy as np
import pandas as pd
from mpi4py import MPI
from sklearn.svm import SVC

import qsvm4eo

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output_name", type=str, default="results")
parser.add_argument(
    "-emb",
    "--embedding_type",
    type=str,
    default="radial",
    choices=["radial", "convolutional"],
    help="Type of embedding used",
)
parser.add_argument("-nfeat", "--num_features", type=int, choices=[4, 8], default=4)
parser.add_argument("-reg", "--regularization", type=float, default=1.0)
parser.add_argument("-conv_sca", "--conv_scaling", type=float, default=37.0)
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



df_train = pd.read_csv("./../data/train_32.csv")
df_test = pd.read_csv("./../data/test_32.csv")
label_names = {
    1: "Urban",
    2: "Agricultural",
    3: "Forests/natural",
}


print("Embedding the data into qubits")
# Embedding the data, transforming the features into qubit coordinates
if args.embedding_type == "radial":
    embedding_train = qsvm4eo.RadialEmbedding(df_train)
    embedding_test = qsvm4eo.RadialEmbedding(df_test)
    qbits_train = embedding_train.embed(shift=1.0, scaling=5.4)
    qbits_test = embedding_test.embed(shift=1.0, scaling=5.4)
    num_features = 4
    y_train = df_train["Label"].values
    y_test = df_test["Label"].values
if args.embedding_type == "convolutional":
    embedding_train = qsvm4eo.ConvolutionalEmbedding(df_train)
    qbits_train, y_train = embedding_train.hsv_embedding(scaling=args.conv_scaling)
    embedding_test = qsvm4eo.ConvolutionalEmbedding(df_test)
    qbits_test, y_test = embedding_test.hsv_embedding(scaling=args.conv_scaling)


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
    "type_of_embedding": args.embedding_type,
    "number_of_features": args.num_features,
    "regularization": args.regularization,
    "convolutional_scaling": args.conv_scaling,
    "train_acc": train_score,
    "test_acc": test_score,
    "labels_pred": y_test_pred.tolist(),
    "labels_true": y_test.tolist() if args.embedding_type == "radial" else y_test,
    "label_names": label_names,
    "time": datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"),
}

with open(f"results_{args.output_name}.json", "w") as fp:
    json.dump(results, fp)
