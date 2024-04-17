import logging
import os.path

import pandas as pd

from news_classification.config.class_corrections import (
    class_corrections,
)


def read_news_article_data(
    path: str, refine_classes: bool = True, n_articles: int = None
) -> pd.DataFrame:
    """
    Reads the news articles from the path provided in the <path> argument.

    :param path:
        absolute path for the json records file
    :param refine_classes:
        Whether classes should be refined (See eda.ipynb)
        This can lead to ~3% increase in (balanced) validation accuracy.
        This is only valid for training data
    :param n_articles:
        Number of article records to be read.
        If None, all records will be read.

    :return:
        Article data as a data-frame
    """
    logging.info(
        f"Reading news article data as dataframe"
        f" from {os.path.basename(path)}"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Path {path} does not exist.")

    articles: pd.DataFrame = pd.read_json(path, lines=True, nrows=n_articles)

    if "headline" not in articles.columns:
        raise ValueError("<headline> column is missing from the data.")

    logging.info(f"Returning {len(articles)} articles")

    if refine_classes and ("category" in articles.columns):
        articles.loc[:, "category"] = articles["category"].replace(
            class_corrections
        )
        logging.info(
            f"With {articles['category'].nunique()} categories after refining."
        )

    return articles
