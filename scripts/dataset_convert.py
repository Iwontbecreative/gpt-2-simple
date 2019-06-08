import pandas as pd
import argparse
from scripts.shared import Separators, Task


def get_args(*in_args):
    parser = argparse.ArgumentParser()

    # === Required parameters === #
    parser.add_argument("--data_file", required=True,
                        type=str, help="tsv to convert")
    parser.add_argument("--output_file", required=True, type=str,
                        help="Where to store the resulting csv")
    parser.add_argument("--task", required=True, type=Task, help="The task name")
    parser.add_argument("--predict_premise", action='store_true',
                        help="Whether to predict the premise or the question")
    args = parser.parse_args(*in_args)
    return args


def convert_task(data: pd.DataFrame, task: Task, predict_premise: bool) -> pd.Series:
    if task == Task.MNLI:
        first_part = data['sentence1']
        second_part = data['sentence2']
        if predict_premise:
            first_part = data['sentence2']
            second_part = data['sentence1']
        data['sentences'] = first_part + Separators.SENT_SEP + second_part + Separators.LABEL_SEP + data["gold_label"]
    elif task == Task.RTE:
        first_part = data['sentence1']
        second_part = data['sentence2']
        if predict_premise:
            first_part = data['sentence2']
            second_part = data['sentence1']
        data['sentences'] = first_part + Separators.SENT_SEP + second_part + Separators.LABEL_SEP + data["label"]
    elif task == Task.COLA:
        if predict_premise:
            print('`predict_premise` does not apply to CoLA')
        return data.iloc[:, -1] + Separators.LABEL_SEP + data.iloc[:, 1].astype(str)
    elif task == Task.WIC:
        if predict_premise:
            print('`predict_premise` does not apply to WiC (tasks are interchangeable)')
            print("This does not handle position yet.")
        first_part = data['sentence1']
        second_part = data['sentence2']
        data['sentences'] = first_part + Separators.SENT_SEP + second_part + Separators.LABEL_SEP + data["label"]
    elif task == Task.COPA:
        if predict_premise:
            print("`predict_premise` is not supported for COPA")
        data['sentences'] = data['premise'] + Separators.SENT_SEP + data['choice1'] + Separators.SENT_SEP + data['choice2'] + Separators.LABEL_SEP + data["label"]
    elif task == Task.WSC:
        if predict_premise:
            print('`predict_premise` does not apply to WSC')
            print("This does not handle position yet.")
        data['sentences'] = data["text"] + Separators.LABEL_SEP + data["label"]
    return data["sentences"]


if __name__ == "__main__":
    args = get_args()
    print("Loading...")
    data_file = args.data_file
    if "json" in data_file:
        data = pd.read_json(data_file)
    else:
        data = pd.read_csv(data_file, sep="\t",
                           error_bad_lines=False, quoting=3, skiprows=0)
    output_series = convert_task(data, args.task, args.predict_premise)
    print("Writing...")
    output_series.to_csv(args.output_file, sep="\t", index=False, header=False)
