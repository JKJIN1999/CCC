#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

module load Python/3.10.4
module load mpi4py/3.1.4
source ~/A1/venvs/venv-3.10.4/bin/activate
pip install argparse
pip install time
mpirun python main.py ~/A1/twitter-100gb.json ~/A1/results/1node1core.txt