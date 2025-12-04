#!/bin/bash -l

#SBATCH -p cpu -N 2
#SBATCH --job-name qc4eo
srun -n 1 python ./run_workflow.py -o $SLURM_JOB_ID -enc radial -reg 1.5 -nfeat 4 -conv_sca 37.0

