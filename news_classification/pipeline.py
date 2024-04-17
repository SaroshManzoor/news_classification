from news_classification.components import (
    apply_categories_to_response_json,
    train_and_evaluate_classifier,
)
from news_classification.config.pipeline_config import PipelineConfig
from news_classification.modelling.classifiers import BaseClassifier
from news_classification.utils import utils

if __name__ == "__main__":
    utils.set_seed()
    utils.init_logging()

    pipeline_config = PipelineConfig.load()

    classifier: BaseClassifier = train_and_evaluate_classifier(
        config=pipeline_config
    )
    classifier.save()

    apply_categories_to_response_json(classifier, config=pipeline_config)
