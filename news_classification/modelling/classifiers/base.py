import logging
import os.path
import warnings
from abc import ABC, abstractmethod
from typing import Iterable

import joblib
import pandas as pd

from news_classification.data_handling.feature_extractors import (
    FeatureExtractor,
    feature_extractor_map,
)
from news_classification.modelling.evaluation import (
    evaluate_classifier,
    generate_report,
)
from news_classification.utils.paths import (
    get_model_registry_path,
)


class BaseClassifier(ABC):
    NAME = ""

    def __init__(
        self,
        feature_extractor: FeatureExtractor = None,
        **kwargs,
    ):
        self.model = None
        self.feature_extractor = feature_extractor
        self.evaluation_metrics: dict = {}

    @abstractmethod
    def train(self, headlines: pd.Series, categories: pd.Series, **kwargs):
        pass

    @abstractmethod
    def predict(self, headlines, **kwargs) -> Iterable:
        pass

    def evaluate(self, x_test, y_test):
        logging.info(f"Evaluating trained {self.NAME}")
        self.evaluation_metrics = evaluate_classifier(self, x_test, y_test)
        generate_report(self.evaluation_metrics, classifier_name=self.NAME)

    def save(self, path=None, **kwargs) -> None:
        if not self.evaluation_metrics:
            warnings.warn("<save> method called before training.")

        if path is None:
            path = os.path.join(get_model_registry_path(), self.NAME)

        os.makedirs(path, exist_ok=True)

        _feature_extractor = (
            self.feature_extractor if self.feature_extractor.PICKLE else None
        )
        pickle_extractor = (
            self.feature_extractor if self.feature_extractor.PICKLE else False
        )

        additional_args = {key: value for key, value in kwargs.items()}

        artifacts = {
            "model": self.model,
            "feature_extractor": _feature_extractor,
            "feature_extractor_name": self.feature_extractor.NAME,
            "pickle_extractor": pickle_extractor,
            "evaluation_metrics": self.evaluation_metrics,
            "additional_args": additional_args,
        }

        logging.info(f"Saving {self.NAME} to model registry.")
        joblib.dump(artifacts, os.path.join(path, f"{self.NAME}.classifier"))

    @classmethod
    def load(cls, path=None, **kwargs) -> "BaseClassifier":
        logging.info(f"Attempting to load pre-trained model from registry.")

        if path is None:
            path = os.path.join(
                get_model_registry_path(), cls.NAME, f"{cls.NAME}.classifier"
            )

            if not os.path.exists(path):
                raise FileNotFoundError("Trained model not found")

        artifacts = joblib.load(filename=path)

        class_instance = cls(**kwargs)
        class_instance.model = artifacts["model"]

        if artifacts["pickle_extractor"]:
            _extractor = artifacts["feature_extractor"]
        else:
            _extractor = feature_extractor_map[
                artifacts["feature_extractor_name"]
            ]()

        class_instance.feature_extractor = _extractor

        for key, value in artifacts["additional_args"].items():
            setattr(class_instance, key, value)

        logging.info(f"Loaded {cls.NAME} successfully.")

        return class_instance
