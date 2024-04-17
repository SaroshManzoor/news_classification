import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from news_classification.utils.data_types import ArrayLike
from news_classification.utils.paths import get_model_registry_path

if TYPE_CHECKING:
    from news_classification.modelling.classifiers import (
        BaseClassifier,
    )


def evaluate_classifier(
    classifier: "BaseClassifier", x_test: ArrayLike, y_test: ArrayLike
) -> dict:
    """
    Returns:
        - accuracy
        - balanced accuracy
        - confusion matrix

    within a dictionary.

    :param classifier:
        classifier to be evaluated
    :param x_test:
        input data
    :param y_test:
        labels

    :return:
        evaluation metrics as a dictionary
    """
    predictions = classifier.predict(x_test)

    if isinstance(predictions, torch.Tensor):
        predictions = predictions

    # noinspection PyUnresolvedReferences
    accuracy = (predictions == y_test).mean() * 100
    logging.info(f"Accuracy: {round(accuracy, 2)}%")

    balanced_accuracy = balanced_accuracy_score(y_test, predictions) * 100
    logging.info(f"Balanced Accuracy: {round(balanced_accuracy, 2)}%")

    class_labels = sorted(np.unique(y_test).tolist())

    _conf_matrix = confusion_matrix(y_test, predictions)

    conf_matrix = pd.DataFrame(
        _conf_matrix,
        index=class_labels,
        columns=class_labels,
    )

    evaluation_metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "confusion_matrix": conf_matrix,
    }

    return evaluation_metrics


def generate_report(metrics: dict, classifier_name: str) -> None:
    """
    Generates an Excel file with two sheets; one for accuracy and the other
    for confusion matrix.
    This file will be stored in storage/model_registry/<classifier_name>
    directory.

    Also saves the heat map of confusion matrix.

    :param metrics:
        evaluation metrics calculated by <evaluation> method
    :param classifier_name:
        name of the classifier

    :return:
    """
    logging.info("Building Report")

    conf_matrix = metrics.pop("confusion_matrix")
    accuracies = pd.Series(metrics)

    classifier_directory = os.path.join(
        get_model_registry_path(), classifier_name
    )
    os.makedirs(classifier_directory, exist_ok=True)

    with pd.ExcelWriter(
        os.path.join(classifier_directory, "report.xlsx")
    ) as report:
        accuracies.to_excel(report, sheet_name="Accuracy")
        conf_matrix.to_excel(report, sheet_name="ConfusionMatrix")

    logging.info("report.xlsx saved in the model registry.")
    from matplotlib import pyplot as plt
    import seaborn

    seaborn.set(font_scale=0.3)

    fig, ax = plt.subplots(dpi=300)
    ax = seaborn.heatmap(
        conf_matrix,
        annot=True,
        ax=ax,
        annot_kws={"fontsize": 3},
        cmap=seaborn.light_palette("seagreen", as_cmap=True),
        cbar=False,
        fmt="g",
    )

    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    plt.xticks(rotation=90)

    plt.savefig(
        os.path.join(classifier_directory, "confusion_matrix.png"),
        bbox_inches="tight",
    )
    plt.close()

    plt.imshow(conf_matrix, cmap="Greens", interpolation="nearest")
