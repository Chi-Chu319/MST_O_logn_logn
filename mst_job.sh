#!/bin/bash
#SBATCH --job-name=mpiTest
#SBATCH --account=project_2009665
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=500M
#SBATCH --output=vertex_generation.out
#SBATCH --partition=small

time srun python3 mst.py 2 2 10
