import pandas as pd
from scripts.shared import Task

LABEL_MAP = {
    "mnli": ["contradiction", "entailment", "neutral"]
}

def get_args(*in_args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", required=True,
                        type=str, help="tsv to convert")
    parser.add_argument("--task", required=True, type=Task, help="The task name")
    args = parser.parse_args(*in_args)
    return args


def main():
    args = get_args()
    task = args.task
    fn = args.input_file
    assert task in LABEL_MAP, f"{task} not supported"
    data = pd.read_csv(fn, sep="\t")
    print(f"Initially {len(data)} rows")
    data = data[data.label.isin(LABEL_MAP)]
    print(f"After label cleaning: {len(data)}")
    data.to_csv(fn, sep="\t", index=False)


if __name__ == "__main__":
    main()