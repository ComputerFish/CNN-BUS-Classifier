#! /bin/bash

#SBATCH --job-name=yourjob
#SBATCH --partition=gpuq-a30
#SBATCH --nodelist=gpu001
#SBATCH --time=5:00:00

source /home/username/miniconda3/bin/activate
conda activate yourenv

srun --unbuffered python -u main.py --version Model_1 --device_name 0