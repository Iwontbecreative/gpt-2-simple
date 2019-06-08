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
    WIC = "wic"
    WSC = "wsc"
