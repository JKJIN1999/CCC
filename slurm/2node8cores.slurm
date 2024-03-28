#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1

module load Python/3.10.4
module load mpi4py/3.1.4
source ~/A1/venvs/venv-3.10.4/bin/activate
pip install argparse
pip install time
mpirun python main.py ~/A1/twitter-100gb.json ~/A1/results/2node8core.txt