import logging
from abc import ABC
from typing import Iterable

import numpy as np


class FeatureExtractor(ABC):
    NAME = ""
    PICKLE = True

    def __init__(self):
        self.transformer = None

    def fit(self, data: Iterable) -> None:
        return self.transformer.fit(data)

    def transform(self, data: Iterable) -> np.ndarray:
        return self.transformer.transform(data)

    def fit_transform(self, data: Iterable) -> np.ndarray:
        logging.info(f"Fitting {self.NAME}")
        return self.transformer.fit_transform(data)
