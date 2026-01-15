# MPI Workflow

In this directory we have a straightforward workflow, involving the following steps:
- Loading the classical Earth Observation data and generating the qubit coordinates from it.
- Running a quantum task where the qubits are evolved with a constant Hamiltonian. The probability distribution of the state of the qubits is return.
- Training a SVM using the computed probability distributions.

## Environment setup
Instead of writing the intermediate data into a file, we initialize MPI environment to send data via high-speed interconnect(via TCP or share memory on VM).

Precondition: install `mpi4py` library with the PATH of ParaStation MPI.
On VM, we can set PATH of installed ParaStation MPI in .bashrc

```bash
PSCOM="/opt/parastation"
export PATH="${PSCOM}/bin${PATH:+:}${PATH}"
export CPATH="${PSCOM}/include${CPATH:+:}${CPATH}"
export LD_LIBRARY_PATH="${PSCOM}/lib64${LD_LIBRARY_PATH:+:}${LD_LIBRARY_PATH}"
export LIBRARY_PATH="${PSCOM}/lib64${LIBRARY_PATH:+:}${LIBRARY_PATH}"

#PSMPI
PARASTATION_MPI="/opt/parastation/mpi"
export PATH="${PARASTATION_MPI}/bin${PATH:+:}${PATH}"
export CPATH="${PARASTATION_MPI}/include${CPATH:+:}${CPATH}"
export LD_LIBRARY_PATH="${PARASTATION_MPI}/lib64${LD_LIBRARY_PATH:+:}${LD_LIBRARY_PATH}"
export LIBRARY_PATH="${PARASTATION_MPI}/lib64${LIBRARY_PATH:+:}${LIBRARY_PATH}"
```

Please set python PATH for the installed mpi4py.

## Usage
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
- `enc` encoding type to use (defaults to `radial`)
- `-nfeat` the number of features to use (either 4 or 8, defaults to 4)
- `-reg` the regularization parameter (defaults to 1.0).
- `conv_sca` Scaling or radius used in convolutional encoding (defaults to 37.0)

The results are saved in a `json` the specified name, in the `run.sh` script we use the `job_id`.
The file contains:
- `train_acc`: the training accuracy
- `test_acc`: the test accuracy
- `labels_pred`: the predicted test set labels
- `labels_true`: the true test set labels
- `time`: the time when the job was completed