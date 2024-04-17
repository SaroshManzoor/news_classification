import yaml
from pydantic import BaseModel

from news_classification.utils.paths import get_config_path


class PipelineConfig(BaseModel):
    """
    Holds all the necessary settings to run the pipeline.

    """

    training_filename: str
    inference_filename: str
    results_filename: str

    # Data Preprocessing
    refine_classes: bool
    preprocess: bool
    lemmatize: bool
    remove_stop_words: bool

    # Feature transformation
    feature_extractor_name: str

    # Classifier
    classifier_name: str

    @classmethod
    def load(cls) -> "PipelineConfig":
        """
        Builds the pydantic model for pipeline configuration based on the
        values in pipeline_config.yml

        pipeline_config.yml is expected to be in <news_classification/config>.

        :return: An instance of PipelineConfig
        """
        config_path = get_config_path()

        with open(config_path) as yml_file:
            config = yaml.safe_load(yml_file)

        return cls(**config)
