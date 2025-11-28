#!/bin/bash -l

#SBATCH -p cpu -N 2
#SBATCH --job-name qc4eo
srun -n 1 python ./run_workflow.py -o $SLURM_JOB_ID -reg 1.5 -nfeat 4

