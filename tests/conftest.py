import os.path

import pytest


@pytest.fixture
def testing_data_path():
    return os.path.join(os.path.dirname(__file__), "testing_data", "news.json")
