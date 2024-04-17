from news_classification.modelling.classifiers.base import (
    BaseClassifier,
)
from news_classification.modelling.classifiers.boosting import (
    BoostingClassifier,
)
from news_classification.modelling.classifiers.custom_mlp import (
    CustomMLPClassifier,
)
from news_classification.modelling.classifiers.sk_mlp import (
    SkMLPClassifier,
)

classifier_map: dict[str:object] = {
    "BoostingClassifier": BoostingClassifier,
    "CustomMLPClassifier": CustomMLPClassifier,
    "SkMLP": SkMLPClassifier,
}
