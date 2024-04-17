from news_classification.data_handling.feature_extractors.base import (
    FeatureExtractor,
)
from news_classification.data_handling.feature_extractors.embeddings import (
    EmbeddingsExtractor,
)
from news_classification.data_handling.feature_extractors.tf_idf import (
    TfIdfExtractor,
)
from news_classification.data_handling.feature_extractors.word_cound import (
    WordCountExtractor,
)

feature_extractor_map: dict[str:object] = {
    "BinaryWordCountExtractor": WordCountExtractor,
    "EmbeddingsExtractor": EmbeddingsExtractor,
    "TfIdfExtractor": TfIdfExtractor,
}
