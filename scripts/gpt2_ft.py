import sys
sys.path.append('.')
import gpt_2_simple as gpt2

model_name = "345M"

sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
              'mnli_nice.tsv',
              model_name=model_name,
              steps=1000)
