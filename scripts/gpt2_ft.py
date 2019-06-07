import gpt_2_simple as gpt2

model_name = "345M"
sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
              'mnli_nice.csv',
              model_name=model_name,
              learning_rate=0.0002,
              save_every=1000,
              sample_length=100,
              batch_size=3,
              steps=100000)
gpt2.generate(sess)