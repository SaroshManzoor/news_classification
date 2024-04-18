import os
import warnings

import news_classification

MAIN_MODULE_PATH = os.path.dirname(news_classification.__file__)
PROJECT_PATH = os.path.dirname(MAIN_MODULE_PATH)
STORAGE_PATH = os.path.join(PROJECT_PATH, "storage")


def get_data_path() -> str:
    """
    Returns the absolute path where training and inference data is supposed
    to reside.

    :return:
    """
    path = os.path.join(PROJECT_PATH, "storage", "data")
    if not os.path.exists(path):
        warnings.warn("Data directory was not found but is now recreated.")

    os.makedirs(path, exist_ok=True)

    return path


def get_model_registry_path() -> str:
    """
    Returns the absolute path for the model registry.
    Model registry stores the trained model objects.

    :return:
        model_registry directory path
    """
    registry_path = os.path.join(PROJECT_PATH, "storage", "model_registry")
    os.makedirs(registry_path, exist_ok=True)

    return registry_path


def get_model_cache_path() -> str:
    """
    Returns the absolute path cache directory.

    The cache is primarily meant for the model downloaded by
    sentence-transformers.

    :return:
        model_registry directory path
    """

    return os.path.join(PROJECT_PATH, ".cache")

def get_config_path() -> str:
    """
    Returns the absolute path for the yaml file containing the necessary
    configurations for the main pipeline

    :return:
    """
    return os.path.join(MAIN_MODULE_PATH, "config", "pipeline_config.yml")
