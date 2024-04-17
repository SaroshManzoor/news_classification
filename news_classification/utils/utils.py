import logging
import os
import sys

import pandas as pd
import torch

from news_classification.utils.constants import SEED


def get_device(gpu=True):
    device_label = "cpu"

    if gpu:
        try:
            if torch.backends.mps.is_available():
                device_label = torch.device("mps")
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        except:
            if device_label == "cpu":
                if torch.cuda.is_available():
                    device_label = torch.device("cuda:0")
                    torch.backends.cudnn.benchmark = True

    return device_label


def init_logging():
    # This is set to disable sentence-transformer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    )

    root_logger.addHandler(stdout_handler)

    logging.getLogger("matplotlib").setLevel(logging.CRITICAL)


def to_percentage(number: float):
    return round(number * 100, 2)


def get_inverse_class_distribution(samples: pd.Series):
    inverse_frequency = 1 / (samples.value_counts(normalize=True))

    return inverse_frequency / inverse_frequency.sum()


def set_seed():
    import numpy
    import random
    import torch

    torch.manual_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)
