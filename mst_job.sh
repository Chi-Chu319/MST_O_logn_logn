#!/bin/bash
#SBATCH --job-name=mpiTest
#SBATCH --account=project_2009665
#SBATCH --time=02:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=mst_result.out
#SBATCH --partition=large

time srun python3 main.py 5 1000 10
