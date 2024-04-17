import os

import pandas as pd

from news_classification.data_handling.preprocessing import (
    preprocess_headlines,
)
from news_classification.data_handling.read_data import (
    read_news_article_data,
)
from news_classification.modelling.classifiers import (
    BoostingClassifier,
)
from news_classification.utils import constants
from news_classification.utils.paths import get_data_path
from news_classification.utils.utils import init_logging

if __name__ == "__main__":
    pipeline_config = {
        "training_filename": constants.TRAINING_DATA_FILENAME,
    }

    init_logging()

    articles: pd.DataFrame = read_news_article_data(
        path=os.path.join(get_data_path(), constants.TRAINING_DATA_FILENAME),
        refine_classes=True,
        n_articles=None,
    )

    articles.loc[:, "headline"] = preprocess_headlines(
        articles["headline"], remove_stop_words=False, lemmatize=False
    )

    classifier = BoostingClassifier()
    classifier.train(
        headlines=articles["headline"], categories=articles["category"]
    )

    classifier.save()

    classifier = BoostingClassifier().load()
    classifier.predict(["This is a headline"])

    # articles: pd.DataFrame = read_news_article_data(
    #     path=os.path.join(get_data_path(), constants.INFERENCE_DATA_FILENAME),
    #     refine=False,
    # )
    # predictions = classifier.predict(
    #     preprocess_headlines(articles["headline"])
    # )
    #
    # articles.loc[:, "category"] = predictions
    # articles.to_json(
    #     os.path.join(get_data_path(), constants.PREDICTIONS_FILENAME),
    #     orient="records",
    #     lines=True,
    # )
