import logging
import os

import pandas as pd

from news_classification.config.pipeline_config import PipelineConfig
from news_classification.data_handling.preprocessing import (
    preprocess_headlines,
)
from news_classification.data_handling.read_data import (
    read_news_article_data,
)
from news_classification.utils.paths import get_data_path


def read_and_preprocess_input(
    config: PipelineConfig, training: bool = True
) -> pd.DataFrame:
    if training:
        filename = config.training_filename
    else:
        filename = config.inference_filename

    articles: pd.DataFrame = read_news_article_data(
        path=os.path.join(get_data_path(), filename)
    )

    if config.preprocess:
        articles.loc[:, "headline"] = preprocess_headlines(
            articles["headline"],
            remove_stop_words=config.remove_stop_words,
            lemmatize=config.lemmatize,
        )
    else:
        logging.info("No preprocessing step specified.")

    return articles
