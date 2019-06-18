import pandas as pd
import glob
import argparse


def get_args(*in_args):
    parser = argparse.ArgumentParser ()

    parser.add_argument("--input_folder", required=True, type=str,
                        help="Where to find the tsvs to collect stats on")
    args = parser.parse_args (*in_args)
    return args


def main():
    args = get_args()
    folder = args.input_folder

    for fn in glob.glob(folder + "/*.tsv"):
        print(f"Handling {fn}")
        data = pd.read_csv(fn, sep="\t")
        if "gold_label" in data.columns:
            print(data["gold_label"].value_counts(True))
        elif "label" in data.columns:
            print(data["label"].value_counts(True))
        else:
            print(f"Could not find label column for {fn}")


if __name__ == "__main__":
    main()