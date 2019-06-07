import gpt_2_simple as gpt2
import pprint

sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess)

pprint.pprint(gpt2.generate(sess, return_as_list=True,
                            truncate="<|endoftext|>", prefix="<|startoftext|>",
                            nsamples=160, batch_size=80))