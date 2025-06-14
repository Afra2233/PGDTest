#!/bin/bash
#SBATCH --job-name=imagenet-val-organize
#SBATCH --account=bdlan08
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=organize_val.%j.out
#SBATCH --error=organize_val.%j.err



module load python/3.8


cd /nobackup/projects/bdlan08/jzhang89/PGDTest

python3 sort_imagenet.py
