import logging
import argparse
from scripts.shared import ModelNames

import gpt_2_simple as gpt2

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def get_args(*in_args):
    parser = argparse.ArgumentParser()

    # === Required parameters === #
    parser.add_argument(
        "--run_name",
        default=None,
        type=str,
        help="Run name to use for training",
    )
    parser.add_argument(
        "--learning_rate",
        default=0.0001,
        type=float,
        help="The LR to use when finetuning the model",
    )
    parser.add_argument(
        "--length",
        default=100,
        type=int,
        help="The maximum length of the samples",
    )
    parser.add_argument(
        "--batch_size",
        default=80,
        type=int,
        help="The batch size to use while generating",
    )
    parser.add_argument(
        "--input_file",
        required=True,
        type=str,
        help="The input file to use for fine-tuning",
    )
    parser.add_argument("--model_name", default=ModelNames.BASE_345, type=str)
    parser.add_argument(
        "--save_every",
        default=1000,
        type=int,
        help="Frequency to save the checkpoint",
    )
    parser.add_argument(
        "--steps",
        default=-1,
        type=int,
        help="The number of steps to finetune for (-1 infinite)",
    )
    args = parser.parse_args(*in_args)
    return args


def main():
    args = get_args()
    sess = gpt2.start_tf_sess()
    input_file = args.input_file
    LOGGER.info(f"Starting finetuning on {input_file}")
    gpt2.finetune(
        sess,
        input_file,
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        save_every=args.save_every,
        sample_length=args.length,
        batch_size=args.batch_size,
        steps=args.steps,
    )
    LOGGER.info("Final model samples")
    gpt2.generate(sess)


if __name__ == "__main__":
    main()
