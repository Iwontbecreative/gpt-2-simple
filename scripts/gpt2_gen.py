import sys
import argparse
from typing import List
sys.path.append('.')

import gpt_2_simple as gpt2
from scripts.shared import Separators
import pandas as pd


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
    samples = [s for s in samples if Separators.SENT_SEP in s and Separators.LABEL_SEP in s]
    return samples


def convert_to_tsv(samples: List[str]) -> pd.DataFrame:
    split_samples = []
    for s in samples:
        sample_dict = {}
        if Separators.SENT_SEP in s:
            first_sent, rest = s.split(Separators.SENT_SEP)
            sample_dict['sentence1'] = first_sent
            second_sent, label = rest.split(Separators.LABEL_SEP)
            sample_dict['sentence2'] = second_sent
            sample_dict['label'] = label
        elif Separators.LABEL_SEP in s:
            first_sent, label = s.split(Separators.LABEL_SEP)
            sample_dict['sentence1'] = first_sent
            sample_dict['label'] = label
        else:
            raise TypeError(f'No separators found in sample {s}')
        split_samples.append(sample_dict)
    return pd.DataFrame(split_samples)


def main():
    args = get_args()
    n_samples = args.n_samples
    batch_size = args.batch_size
    # Avoid cases where `n_samples` is not divisible by batch size
    if n_samples % batch_size:
        n_samples = batch_size * (n_samples // batch_size)

    sess = gpt2.start_tf_sess()
    if args.run_name:
        gpt2.load_gpt2(sess, args.run_name)
    else:
        gpt2.load_gpt2(sess)
    example = ""
    if args.example:
        example = args.example + Separators.SENT_SEP
    print("Generating samples...")
    samples = gpt2.generate(sess, return_as_list=True,
                            truncate=Separators.EOS, prefix=Separators.BOS + example,
                            nsamples=n_samples, batch_size=batch_size,
                            length=args.length)
    samples = filter_bad_samples(samples)
    print(f"Generated {len(samples)} correct samples")
    print("Preview:")
    for i in samples[:10]:
        print(i, "\n")
    samples_df = convert_to_tsv(samples)
    output_file = args.output_file
    print(f"Writing samples to {output_file}")
    samples_df.to_csv(output_file, sep="\t", index=False)


if __name__ == "__main__":
    main()


