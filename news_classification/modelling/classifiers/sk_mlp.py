import logging
from typing import Iterable, List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from news_classification.data_handling.feature_extractors import (
    FeatureExtractor,
)
from news_classification.modelling.classifiers.base import (
    BaseClassifier,
)
from news_classification.utils.constants import SEED
from news_classification.utils.utils import to_percentage


class SkMLPClassifier(BaseClassifier):
    """
    SK-Learn MLP using binary word counts as inputs
    """

    NAME = "SkMLPClassifier"

    def __init__(self, feature_extractor: FeatureExtractor, **kwargs):
        BaseClassifier.__init__(
            self,
            feature_extractor,
            **kwargs,
        )

        self.feature_extractor = feature_extractor
        self.model = MLPClassifier(
            random_state=SEED,
            max_iter=20,
            batch_size=512,
            hidden_layer_sizes=5,
            learning_rate_init=0.005,
            verbose=True,
            early_stopping=True,
        )

        self.test_size = 0.2

    def train(
        self, headlines: pd.Series, categories: pd.Series, **kwargs
    ) -> None:
        _test_size = round(self.test_size * 100)

        logging.info(
            f"Splitting training data with a "
            f"{100 - _test_size}-{_test_size} split."
        )
        x_train, x_test, training_labels, test_labels = train_test_split(
            headlines,
            categories,
            test_size=self.test_size,
            random_state=0,
            stratify=categories,
        )

        x_train = x_train[x_train.apply(lambda x: len(x.split())) >= 3]
        training_labels = training_labels.loc[x_train.index]

        features = self.feature_extractor.fit_transform(x_train)

        logging.info(f"Beginning training of {self.NAME}")
        checkpoint = pd.Timestamp.now()
        self.model.fit(features, training_labels)

        logging.info(
            f"Training took {(pd.Timestamp.now() - checkpoint).seconds}s"
        )

        logging.info(
            f"Training set accuracy: "
            f"{to_percentage((self.predict(features) == training_labels).mean())}"
        )

        self.evaluate(x_test, test_labels)

    def predict(self, headlines: Iterable, **kwargs) -> List[str]:
        return self.model.predict(self.feature_extractor.transform(headlines))

    def save(self, *args, **kwargs) -> None:
        pass

    def load(self, *args, **kwargs) -> "SkMLPClassifier":
        pass
