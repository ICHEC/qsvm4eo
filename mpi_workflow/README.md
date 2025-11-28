# MPI Workflow

In this directory we have a straightforward workflow, involving the following steps:
- Loading the classical Earth Observation data and generating the qubit coordinates from it.
- Running a quantum task where the qubits are evolved with a constant Hamiltonian. The probability distribution of the state of the qubits is return.
- Training a SVM using the computed probability distributions.

It can either be run as a batched job with
```
sbatch run.sh
```
or interactively with
```
salloc -N2 --partition=cpu
srun -n 1 python run_workflow.py
```

The `run_workflow.py` script takes the following optional arguemnts
- `-o` the name of the output file (defaults to `results`)
- `-nfeat` the number of features to use (either 4 or 8, defaults to 4)
- `-reg` the regularization parameter (defaults to 1.0).

The results are saved in a `json` the specified name, in the `run.sh` script we use the `job_id`.
The file contains:
- `train_acc`: the training accuracy
- `test_acc`: the test accuracy
- `labels_pred`: the predicted test set labels
- `labels_true`: the true test set labels
- `time`: the time when the job was completed