#!/bin/bash


source activate gpt2
CODE_PATH=/Users/thibault/Code/gpt-2-simple
GLUE_CONVERTED=/Users/thibault/Code/gpt-2-simple/data/converted_glue/
PYTHONPATH=${CODE_PATH}
PATH=${PYTHONPATH}:$PATH

#cd ${CODE_PATH}

TASK=mnli
LENGTH=50
N_SAMPLES=10
SEED=`shuf -i1000-9999 -n1`
COND_GEN=${GLUE_CONVERTED}/ultra_small_mnli.csv


python scripts/gpt2_gen.py --output_file ${TASK}_conditional_samples_${SEED}.tsv --n_samples ${N_SAMPLES} --length ${LENGTH} --batch_size 3 --run_name ${TASK}_small --task ${TASK}  --conditional_gen_file ${COND_GEN}
echo DONE
