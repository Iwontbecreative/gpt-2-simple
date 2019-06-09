import argparse

import pandas as pd
from scripts.shared import Task

LABEL_MAP = {
    "mnli": ["contradiction", "entailment", "neutral"],
    "rte": ["entailment", "not_entailment"],
    "cola": [0, 1],
    "copa": [0, 1],
}


def get_args(*in_args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file", required=True, type=str, help="tsv to convert"
    )
    parser.add_argument(
        "--task", required=True, type=Task, help="The task name"
    )
    args = parser.parse_args(*in_args)
    return args


def mnli_fix(df: pd.DataFrame) -> pd.DataFrame:
    df.rename({"label": "gold_label"}, axis=1, inplace=True)
    # Create missing columns
    cols = ["index", "promptID", "pairID", "genre", "sentence1_binary_parse",
            "sentence2_binary_parse", "sentence1_parse", "sentence2_parse",
            "sentence1", "sentence2", "label1", "gold_label"]
    for col in cols:
        if col not in df:
            df[col] = "filler"
    # Re-order columns
    df = df[cols]
    return df


def cola_fix(df: pd.DataFrame) -> pd.DataFrame:
    df['genre'] = "gj04"
    df['star'] = df.label.apply(lambda s: "*" if s else "")
    return df[['genre', 'label', 'star', 'sentence1']]


def rte_fix(df: pd.DataFrame) -> pd.DataFrame:
    df['index'] = np.arange(len(df))
    return df[['index', 'sentence1', 'sentence2', 'label']]


def main():
    args = get_args()
    task = args.task
    fn = args.input_file
    assert task.value in LABEL_MAP, f"{task} not supported"
    authorized_labels = LABEL_MAP[task.value]
    data = pd.read_csv(fn, sep="\t")
    print(f"Initially {len(data)} rows")
    data = data[data.label.isin(authorized_labels)]
    print(f"After label cleaning: {len(data)}")
    if task == Task.MNLI:
        data = mnli_fix(data)
    elif task == Task.COLA
        data = cola_fix(data)
    elif task == Task.RTE:
        data = rte_fix(data)

    print(f"After dataset specific fixes: {len(data)}")
    header = task != Task.COLA
    data.to_csv(fn, sep="\t", index=False, header=header)


if __name__ == "__main__":
    main()
