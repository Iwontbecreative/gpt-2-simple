import pandas as pd

MNLI_PATH = "/scratch/tjf324/data/glue_auto_dl/MNLI/train.tsv"

train = pd.read_csv(MNLI_PATH, sep="\t", error_bad_lines=False, quoting=3, skiprows=0)

train['sentences'] = train['sentence1'] + ' | ' +train['sentence2'] + " || " + train["gold_label"]

train["sentences"].to_csv("mnli_nice.csv", sep="\t", index=False, header=False)

