import pandas as pd
from scripts.shared import Task, LABEL_MAP


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
    df['star'] = df.label.apply(lambda s: "" if s else "*")
    return df[['genre', 'label', 'star', 'sentence1']]


def rte_fix(df: pd.DataFrame) -> pd.DataFrame:
    df['index'] = list(range(len(df)))
    return df[['index', 'sentence1', 'sentence2', 'label']]


def task_fixes(df: pd.DataFrame, task: Task) -> pd.DataFrame:
    assert task.value in LABEL_MAP, f"{task} not supported"
    authorized_labels = LABEL_MAP[task.value]
    df = df[df.label.isin(authorized_labels)]
    if task == Task.MNLI:
        df = mnli_fix(df)
    elif task == Task.COLA:
        df = cola_fix(df)
    elif task == Task.RTE:
        df = rte_fix(df)
    else:
        raise NotImplementedError(f'No handling defined for task {task}')
    return df
