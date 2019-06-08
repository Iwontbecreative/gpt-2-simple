from enum import Enum


class Separators(Enum):
    LABEL_SEP = " || "
    SENT_SEP = " | "
    BOS = "<|startoftext|>"
    EOS = "<|endoftext|>"