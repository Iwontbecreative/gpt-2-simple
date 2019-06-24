#!/bin/bash

# Generic job script for all experiments on NYU CILVR machines.
#SBATCH --output=/scratch/tjf324/slurm/gen_gpt2_%j.out
#SBATCH --gres=gpu:p40:1
#SBATCH --mem=16000
#SBATCH --time=24:00:00

CODE_PATH=/scratch/tjf324/gpt-2-simple

PATH=/scratch/tjf324/anaconda/envs/gpt2/bin:$PATH
PYTHONPATH=${CODE_PATH}

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` - $JIANT_OVERRIDES >> ~/gpt2_machine_assignments.txt


source activate gpt2
# Run.
cd ${CODE_PATH}

TASK=mnli
LENGTH=100
N_SAMPLES=10
SEED=`shuf -i1000-9999 -n1`
COND_GEN=/scratch/tjf324/data/converted_glue/ultra_small_mnli.csv

python scripts/gpt2_gen.py --output_file ${TASK}_conditional_samples_${SEED}.tsv --n_samples ${N_SAMPLES} --length ${LENGTH} --batch_size 10 --run_name ${TASK} --task ${TASK}  --conditional_gen_file ${COND_GEN}
echo DONE
