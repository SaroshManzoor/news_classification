import logging
import os

import pandas as pd

from news_classification.components.data import (
    read_and_preprocess_input,
)
from news_classification.config.pipeline_config import PipelineConfig
from news_classification.data_handling.preprocessing import (
    preprocess_headlines,
)
from news_classification.modelling.classifiers import (
    BaseClassifier,
)
from news_classification.utils.paths import get_data_path


def apply_categories_to_response_json(
    trained_classifier: BaseClassifier, config: PipelineConfig
):
    articles: pd.DataFrame = read_and_preprocess_input(config, training=False)

    logging.info(f"Predicting categories for {config.inference_filename}")
    predictions = trained_classifier.predict(
        preprocess_headlines(articles["headline"])
    )

    articles.loc[:, "category"] = predictions

    results_path = os.path.join(get_data_path(), config.results_filename)
    logging.info(f"Saving results to {results_path}")

    articles.to_json(
        results_path,
        orient="records",
        lines=True,
    )
