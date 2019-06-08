import sys
from typing import List
sys.path.append('.')

import gpt_2_simple as gpt2
import pickle as pkl
import pprint


def get_args(*in_args):
    parser = argparse.ArgumentParser()

    # === Required parameters === #
    parser.add_argument("--run_name", default=None, type=str,
                        help="Run name to use for generation")
    parser.add_argument("--output_file", required=True, type=str,
                        help="Where to store the resulting csv")
    parser.add_argument("--example", default="", type=str,
                        help="Starter example to be added to the prefix. E.g MNLI text.")
    parser.add_argument("--n_samples", default=1000, type=int,
                        help="The number of samples to generate")
    parser.add_argument("--length", default=100, type=int,
                        help="The maximum lenght of the samples")
    parser.add_argument("--batch_size", default=80, type=int,
                        help="The batch size to use while generating")
    args = parser.parse_args(*in_args)
    return args


def filter_bad_samples(samples: List[str]) -> List[str]:
    """
    Filter generations which do not respect the codes
    """
    samples = [s for s in samples if " | " in s and " || " in s]
    return samples


def main():
    args = get_args()
    sess = gpt2.start_tf_sess()
    if args.run_name:
        gpt2.load_gpt2(sess, args.run_name)
    else:
        gpt2.load_gpt2(sess)
    example = ""
    if args.example:
        example = args.example + " | "
    samples = gpt2.generate(sess, return_as_list=True,
                            truncate="<|endoftext|>", prefix="<|startoftext|>" + example,
                            nsamples=args.n_samples, batch_size=args.batch_size,
                            length=args.length)
    samples = filter_bad_samples(samples)
    with open(args.output_file, 'wb') as outfile:
        pickle.dump(samples, outfile)
    print(f"Generated {len(samples)} correct samples")
    for i in samples[:10]:
        print(i, "\n")





if __name__ == "__main__":
    main()


