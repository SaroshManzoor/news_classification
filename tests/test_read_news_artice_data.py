import pandas as pd

from news_classification.data_handling.read_data import read_news_article_data


def test_read_news_article_return_type(testing_data_path):
    assert isinstance(read_news_article_data(testing_data_path), pd.DataFrame)
