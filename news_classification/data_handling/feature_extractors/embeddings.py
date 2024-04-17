import logging
from typing import Iterable

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from news_classification.data_handling.feature_extractors.base import (
    FeatureExtractor,
)
from news_classification.utils.utils import get_device


class EmbeddingsExtractor(FeatureExtractor):
    NAME = "EmbeddingsExtractor"
    PICKLE = False

    def __init__(self):
        super().__init__()

        self.transformer = SentenceTransformer(
            "all-MiniLM-L6-v2", device=get_device(), cache_folder="./.cache"
        )

    def fit(self, data: Iterable) -> None:
        pass

    def transform(self, data: Iterable) -> np.ndarray:
        logging.info("Getting embeddings for headlines.")
        _sentences = data

        if isinstance(_sentences, str):
            _sentences = [_sentences]

        if isinstance(_sentences, (pd.Series, np.ndarray)):
            _sentences = _sentences.tolist()

        return self.transformer.encode(
            _sentences,
            normalize_embeddings=True,
            batch_size=1024,
        )

    def fit_transform(self, data: Iterable) -> np.ndarray:
        return self.transform(data)
