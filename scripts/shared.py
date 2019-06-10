from enum import Enum


class Separators:
    LABEL_SEP = " || "
    SENT_SEP = " | "
    BOS = "<|startoftext|>"
    EOS = "<|endoftext|>"


class ModelNames:
    BASE_345 = "345M"
    BASE_117 = "117M"


class Task(Enum):
    MNLI = "mnli"
    RTE = "rte"
    COLA = "cola"
    COPA = "copa"
    WIC = "wic"
    WSC = "wsc"


LABEL_MAP = {
    "mnli": ["contradiction", "entailment", "neutral"],
    "rte": ["entailment", "not_entailment"],
    "cola": ["0", "1"],
    "copa": ["0", "1"],
    "sst": ["0", "1"],
    "mrpc": ["0", "1"],
}
