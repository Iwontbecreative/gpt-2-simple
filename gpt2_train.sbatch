#!/bin/bash

# Generic job script for all experiments on NYU CILVR machines.
#SBATCH --output=/scratch/tjf324/slurm/gpt2_%j.out
#SBATCH --gres=gpu:p40:1
#SBATCH --mem=16000
#SBATCH --time=24:00:00

CODE_PATH=/scratch/tjf324/gpt-2-simple

PATH=/scratch/tjf324/anaconda/envs/gpt2/bin:/scrathc/tjf324/anaconda3/bin:$PATH
PYTHONPATH=${CODE_PATH}

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` - $JIANT_OVERRIDES >> ~/gpt2_machine_assignments.txt

conda activate gpt2
# Run.
cd ${CODE_PATH}



python scripts/gpt2_ft.py
echo DONE
