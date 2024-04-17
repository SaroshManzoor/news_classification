import pandas as pd

from news_classification.components.data import (
    read_and_preprocess_input,
)
from news_classification.config.pipeline_config import PipelineConfig
from news_classification.data_handling.feature_extractors import (
    feature_extractor_map,
)
from news_classification.modelling.classifiers import (
    BaseClassifier,
    classifier_map,
)


def train_and_evaluate_classifier(config: PipelineConfig) -> BaseClassifier:
    articles: pd.DataFrame = read_and_preprocess_input(config, training=True)

    classifier = _init_classifier(config)

    classifier.train(
        headlines=articles["headline"],
        categories=articles["category"],
    )

    return classifier


def _init_classifier(config: PipelineConfig) -> BaseClassifier:
    _classifier = classifier_map[config.classifier_name]
    _feature_extractor = feature_extractor_map[config.feature_extractor_name]

    return _classifier(feature_extractor=_feature_extractor())
