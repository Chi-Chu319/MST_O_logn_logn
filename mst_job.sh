#!/bin/bash
#SBATCH --job-name=mpiTest
#SBATCH --account=project_2009665
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=16G
#SBATCH --output=mst_result.out
#SBATCH --partition=large

time srun python3 main.py 8192 1000 10
