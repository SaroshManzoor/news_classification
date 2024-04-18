import logging
from typing import Iterable, List

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from news_classification.data_handling.feature_extractors import (
    FeatureExtractor,
)
from news_classification.modelling.classifiers.base import (
    BaseClassifier,
)
from news_classification.utils.constants import SEED
from news_classification.utils.utils import to_percentage


class LogisticRegressionClassifier(BaseClassifier):
    NAME = "LogisticRegressionClassifier"

    def __init__(self, feature_extractor: FeatureExtractor, **kwargs):
        BaseClassifier.__init__(
            self,
            feature_extractor,
            **kwargs,
        )

        self.model = LogisticRegression(
            n_jobs=-1,
            class_weight="balanced",
            random_state=SEED,
            C=0.5
        )

        self.test_size = 0.2

    def train(
        self, headlines: pd.Series, categories: pd.Series, **kwargs
    ) -> None:
        """
        See parent class's (BaseClassifier) docstring for details

        :param headlines:
        :param categories:
        :param kwargs:
        :return:
        """
        _test_size = round(self.test_size * 100)

        logging.info(
            f"Splitting training data with a "
            f"{100 - _test_size}-{_test_size} split."
        )
        x_train, x_test, y_train, y_test = train_test_split(
            headlines,
            categories,
            test_size=self.test_size,
            random_state=0,
            stratify=categories,
        )

        # Use headlines with at least 2 words for training
        x_train = x_train[x_train.apply(lambda x: len(x.split())) >= 2]
        y_train = y_train.loc[x_train.index]

        features = self.feature_extractor.fit_transform(x_train)

        logging.info(f"Beginning training of {self.NAME}")
        checkpoint = pd.Timestamp.now()
        self.model.fit(features, y_train)

        logging.info(
            f"Training took {(pd.Timestamp.now() - checkpoint).seconds}s"
        )

        logging.info(
            f"Training set accuracy: "
            f"{to_percentage((self.predict(x_train) == y_train).mean())}"
        )

        self.evaluate(x_test, y_test)

    def predict(self, headlines: Iterable, **kwargs) -> List[str]:
        return self.model.predict(self.feature_extractor.transform(headlines))
