from enum import Enum
import logging

logging_config = dict(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


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


CHECKPOINT_DIR = "checkpoint"
SAMPLE_DIR = "samples"
