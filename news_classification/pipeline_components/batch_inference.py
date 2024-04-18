import logging
import os

import pandas as pd

from news_classification.config.pipeline_config import PipelineConfig
from news_classification.modelling.classifiers import (
    BaseClassifier,
)
from news_classification.pipeline_components.data import (
    read_and_preprocess_input,
)
from news_classification.utils.paths import get_data_path


def apply_categories_to_response_json(
    trained_classifier: BaseClassifier, config: PipelineConfig
) -> None:
    """
    Uses the <trained_classifier> to infer categories of the articles
    in json file specified by "inference_filename" argument of
    pipeline configuration.

    The resul is then stored back in storage/data directory with the name
    specified in "result" argument of pipeline configuration

    :param trained_classifier:
        BaseClassifier
    :param config:
        PipelineConfig

    :return:
    """
    articles: pd.DataFrame = read_and_preprocess_input(config, training=False)

    logging.info(f"Predicting categories for {config.inference_filename}")

    predictions = trained_classifier.predict(articles["headline"])
    articles.loc[:, "category"] = predictions

    results_path = os.path.join(get_data_path(), config.results_filename)
    logging.info(f"Saving results to {results_path}")

    articles.to_json(
        results_path,
        orient="records",
        lines=True,
    )
