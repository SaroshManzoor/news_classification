from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from news_classification.data_handling.feature_extractors.base import (
    FeatureExtractor,
)


class TfIdfExtractor(FeatureExtractor):
    NAME = "TfIdfExtractor"

    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize

        # setting min_df/max_features to a higher value could result
        # in memory issues, however will improve classifier performance.
        _tf_idf = TfidfVectorizer(
            min_df=1e-3, stop_words="english", use_idf=True
        )

        self.transformer = _tf_idf
        self.scaler = StandardScaler()

    def fit_transform(self, data: Iterable) -> np.ndarray:
        self.fit(data)

        return self.transform(data)

    def fit(self, data: Iterable) -> None:
        _data = self.transformer.fit_transform(data).toarray()

        if self.normalize:
            self.scaler.fit(_data)

    def transform(self, data: Iterable) -> np.ndarray:
        data = self.transformer.transform(data).toarray()

        if self.normalize:
            data = self.scaler.transform(data)

        return data
