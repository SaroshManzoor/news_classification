import logging

from imblearn.over_sampling import SMOTE

from news_classification.utils.constants import SEED


def oversample(features, training_labels):
    logging.info("Oversampling minority classes.")
    over_sampler = SMOTE(random_state=SEED, k_neighbors=2)

    features, training_labels = over_sampler.fit_resample(
        features, training_labels
    )

    logging.info("Oversampling complete")

    return features, training_labels
