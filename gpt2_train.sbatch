#!/bin/bash

# Generic job script for all experiments on NYU CILVR machines.
#SBATCH --output=/scratch/tjf324/slurm/gpt2_%j.out
#SBATCH --gres=gpu:p40:1
#SBATCH --mem=16000
#SBATCH --time=24:00:00

CODE_PATH=/scratch/tjf324/gpt-2-simple
GLUE_CONVERTED=/scratch/tjf324/data/converted_glue/
PATH=/scratch/tjf324/anaconda/envs/gpt2/bin:/scratch/tjf324/anaconda3/bin:$PATH
PYTHONPATH=${CODE_PATH}

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` - $JIANT_OVERRIDES >> ~/gpt2_machine_assignments.txt

source activate gpt2
# Run.
cd ${CODE_PATH}



python scripts/gpt2_ft.py --run_name mnli_small --max_checkpoints 10 --save_every 500 --length 200 --batch_size 3 --input_file ${GLUE_CONVERTED}/mnli.csv --steps 5000
echo DONE
