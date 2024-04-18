from news_classification import pipeline_components as components
from news_classification.config.pipeline_config import PipelineConfig
from news_classification.modelling.classifiers import BaseClassifier
from news_classification.utils import utils

if __name__ == "__main__":
    utils.set_seed()
    utils.init_logging()

    pipeline_config = PipelineConfig.load()

    classifier: BaseClassifier = components.train_and_evaluate_classifier(
        config=pipeline_config
    )
    classifier.save()

    components.apply_categories_to_response_json(
        classifier, config=pipeline_config
    )
