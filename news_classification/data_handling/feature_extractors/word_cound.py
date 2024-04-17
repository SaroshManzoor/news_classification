import logging
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer

from news_classification.data_handling.feature_extractors.base import (
    FeatureExtractor,
)


class WordCountExtractor(FeatureExtractor):
    NAME = "BinaryWordCountExtractor"

    def __init__(self, binary=True):
        super().__init__()
        self.binary = binary

        self.transformer = make_pipeline(
            CountVectorizer(min_df=1e-5, stop_words="english")
        )

        self.binarizer = Binarizer()

    def fit_transform(self, data: Iterable) -> np.ndarray:
        logging.info(f"Fitting {self.NAME}")
        self.fit(data)

        return self.transform(data)

    def fit(self, data: Iterable) -> None:
        _data = self.transformer.fit_transform(data).toarray()

        if self.binary:
            self.binarizer.fit(_data)

    def transform(self, data: Iterable) -> np.ndarray:
        data = self.transformer.transform(data).toarray()

        if self.binary:
            data = self.binarizer.transform(data)

        return data
