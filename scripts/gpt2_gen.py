import sys
import argparse
import logging
import os
from typing import List

sys.path.append(".")

import tqdm

import gpt_2_simple as gpt2
from scripts.shared import Separators, Task, logging_config
from scripts.task_format_fixes import task_fixes
import pandas as pd

logging.basicConfig(**logging_config)
LOGGER = logging.getLogger(__name__)


def get_args(*in_args):
    parser = argparse.ArgumentParser()

    # === Required parameters === #
    parser.add_argument(
        "--run_name",
        default=None,
        type=str,
        help="Run name to use for generation",
    )
    parser.add_argument(
        "--conditional_gen_file",
        default=None,
        type=str,
        help="If given, will perform conditional generation based on examples of the file"
    )
    parser.add_argument(
        "--output_file",
        required=True,
        type=str,
        help="Where to store the resulting csv",
    )
    parser.add_argument(
        "--example",
        default="",
        type=str,
        help="Starter example to be added to the prefix. E.g MNLI text.",
    )
    parser.add_argument(
        "--n_samples",
        default=1000,
        type=int,
        help="The number of samples to generate",
    )
    parser.add_argument(
        "--length",
        default=100,
        type=int,
        help="The maximum length of the samples",
    )
    parser.add_argument(
        "--task",
        required=True,
        type=Task,
        help="Which task to generate for (for post-processing handling",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature to use when sampling",
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0,
        help="Top_p to use when sampling"
    )
    parser.add_argument(
        "--batch_size",
        default=100,
        type=int,
        help="The batch size to use while generating",
    )
    args = parser.parse_args(*in_args)
    return args


def filter_bad_samples(samples: List[str], task: Task) -> List[str]:
    """
    Filter generations which do not respect the codes
    """
    samples = [s for s in samples if Separators.LABEL_SEP in s]
    if task in ["mnli", "rte", "snli", "copa"]:
        samples = [
            s
            for s in samples
            if Separators.SENT_SEP in s and Separators.LABEL_SEP in s
        ]
    return samples


def remove_delimiter_artefacts(s: str) -> str:
    return str(s).split("<")[0]


def convert_to_tsv(samples: List[str]) -> pd.DataFrame:
    # FIXME: Check labels are in allowed values?
    split_samples = []
    for s in samples:
        sample_dict = {}
        if Separators.SENT_SEP in s and Separators.LABEL_SEP in s:
            first_sent, *rest = s.split(Separators.SENT_SEP)
            if len(rest) != 1:
                LOGGER.warning("Faulty example: %s", s)
                continue
            rest = rest[0]
            sample_dict["sentence1"] = first_sent
            second_sent, *label = rest.split(Separators.LABEL_SEP)
            if len(label) != 1:
                LOGGER.warning("Faulty example: %s", s)
                continue
            label = label[0]
            sample_dict["sentence2"] = second_sent
            sample_dict["label"] = label
        elif Separators.LABEL_SEP in s:
            first_sent, *label = s.split(Separators.LABEL_SEP)
            if len(label) != 1:
                LOGGER.warning("Faulty example: %s", s)
                continue
            label = label[0]
            sample_dict["sentence1"] = first_sent
            sample_dict["label"] = label
        else:
            raise TypeError(f"No separators found in sample {s}")
        split_samples.append(sample_dict)
    df = pd.DataFrame(split_samples)
    # Remove some artefacts, not all sadly...
    df["label"] = df["label"].apply(remove_delimiter_artefacts)
    return df


def main() -> None:
    args = get_args()
    n_samples = args.n_samples
    batch_size = args.batch_size
    # Avoid cases where `n_samples` is not divisible by `batch size`
    if n_samples % batch_size:
        n_samples = batch_size * (n_samples // batch_size)

    sess = gpt2.start_tf_sess()
    run_name = args.run_name
    gpt2.load_gpt2(sess, args.run_name)

    conditional_gen_file = args.conditional_gen_file
    if conditional_gen_file is None:
        example = ""
        if args.example:
            example = args.example + Separators.SENT_SEP
        LOGGER.info("Generating samples...")
        samples = gpt2.generate(
            sess,
            return_as_list=True,
            temperature=args.temperature,
            top_p=args.top_p,
            truncate=Separators.EOS,
            prefix=Separators.BOS + example,
            nsamples=n_samples,
            batch_size=batch_size,
            run_name=run_name,
            length=args.length,
        )
    else:
        LOGGER.info("Generating conditional samples...")
        assert os.path.exists(conditional_gen_file)
        samples = []
        with open(conditional_gen_file, "r") as infile:
            for line in tqdm.tqdm(infile, desc="Generating samples for example"):
                line = line.split(" | ")[0]
                example = Separators.BOS + line + Separators.EOS
                samples.extend(gpt2.generate(
                    sess,
                    return_as_list=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    truncate=Separators.EOS,
                    prefix=Separators.BOS + example,
                    nsamples=n_samples,
                    batch_size=batch_size,
                    run_name=run_name,
                    length=args.length,
                ))



    task = args.task
    LOGGER.info("Originally had %s", len(samples))
    samples = filter_bad_samples(samples, task)
    samples = [
        s.replace(Separators.BOS, "").replace("\n", " ").replace("\t", " ")
        for s in samples
    ]
    LOGGER.info("Generated %s splittable samples", len(samples))
    LOGGER.info("Preview:")
    for i in samples[:10]:
        LOGGER.info(i + "\n")
    samples_df = convert_to_tsv(samples)
    LOGGER.info(
        "After additional checks, generated %s samples", len(samples_df)
    )
    samples_df.drop_duplicates(inplace=True)
    LOGGER.info(
        "After removing duplicates, generated %s samples", len(samples_df)
    )
    output_file = args.output_file
    LOGGER.info(f"Distribution of labels:")
    top_distr = samples_df["label"].value_counts(True).head(5) * 100
    LOGGER.info("Label distribution")
    LOGGER.info(top_distr)
    LOGGER.info("Applying fixes specific to %s", task)
    samples_df = task_fixes(samples_df, task)

    LOGGER.info(f"Writing samples to %s", output_file)
    header = task != Task.COLA
    samples_df.to_csv(output_file, sep="\t", index=False, header=header)


if __name__ == "__main__":
    main()
