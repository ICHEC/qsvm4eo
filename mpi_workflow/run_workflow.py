import qsvm4eo
import numpy as np
import json
import datetime
import pandas as pd 
from mpi4py import MPI
from sklearn.svm import SVC
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_type', type=str, default='radial', help="Type of encoding used")
encoding_type = parser.parse_args().encoding_type



def compute_probabilities(qbits):
    # Create a worker for the quantum task
    worker = MPI.COMM_SELF.Spawn("python", "./qsvm_train.py", 1)

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
    return probs


# Load the data
if encoding_type == 'radial':
    num_features = 4
    x_train, y_train, x_test, y_test = qsvm4eo.load_data(
        data_path="..", num_features=num_features, scale_features=False
    )
if encoding_type == 'convolutional':
    df_train = pd.read_csv("./../data/train_32.csv")
    df_test = pd.read_csv("./../data/test_32.csv")




# Encode the data, transforming the features into qubit coordinates
if encoding_type == 'radial':
    encoding = qsvm4eo.RadialEncoding(
        max_feature=np.max(x_train), shift=1.0, scaling=5.4, n_features=num_features
    )
    qbits_train = [encoding.encode(x) for x in x_train]
    qbits_test = [encoding.encode(x) for x in x_test]
if encoding_type == 'convolutional':
    encoding_train = qsvm4eo.ConvolutionalEncoding(
        df_train
    )
    qbits_train, y_train = encoding_train.hsv_encoding(scaling=37)
    encoding_test = qsvm4eo.ConvolutionalEncoding(
        df_test
    )
    qbits_test, y_test = encoding_test.hsv_encoding(scaling=37)
    



print("Qubit Geometries:")
print(qbits_train[:4])


print("Training:")
probs_train = compute_probabilities(qbits_train)
excitations_train = qsvm4eo.compute_excitation_count(probs_train)

# Compute the kernel
kernel = qsvm4eo.Kernel()
gram_train = kernel.compute_gram_train(excitations_train)

# Fit the SVM ad get the score
model = SVC(kernel="precomputed")
model.fit(gram_train, y_train)
train_score = model.score(gram_train, y_train)

print("Testing")
probs_test = compute_probabilities(qbits_test)
excitations_test = qsvm4eo.compute_excitation_count(probs_test)

# Compute the kernel and score
gram_test = kernel.compute_gram_train(excitations_test)
y_test_pred = model.predict(gram_test)
test_score = model.score(gram_test, y_test)

print("Train acc:", train_score)
print("Test acc:", test_score)

results = {
    "train_acc": train_score,
    "test_acc": test_score,
    "y_test_pred": y_test_pred.tolist(),
    "time": datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"),
}

with open("results.json", "w") as fp:
    json.dump(results, fp)

    
