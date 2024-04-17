import logging
from typing import Iterable

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from news_classification.data_handling.custom_dataset import (
    CustomDataset,
)
from news_classification.data_handling.feature_extractors import (
    FeatureExtractor,
)
from news_classification.modelling.classifiers._custom_mlp import (
    CustomMLPSmall,
)
from news_classification.modelling.classifiers.base import (
    BaseClassifier,
)
from news_classification.utils import utils
from news_classification.utils.constants import SEED


class CustomMLPClassifier(BaseClassifier):
    NAME = "CustomMLPClassifier"

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        device=utils.get_device(),
    ):

        BaseClassifier.__init__(self, feature_extractor=feature_extractor)

        # The underlying model will be initialized after number of inputs
        # to the model has been determined in the training step
        self.model = None
        self.test_size = 0.2
        self.device = device
        self.label_to_class_index: dict = {}
        self.class_index_to_label: dict = {}

    def train(self, headlines, categories, **kwargs):
        batch_size = 2048
        n_epochs = 10

        self.create_label_to_index_map(categories)
        _test_size = round(self.test_size * 100)

        logging.info(
            f"Splitting training data with a "
            f"{100 - _test_size}-{_test_size} split."
        )

        x_train, x_test, y_train, y_test = train_test_split(
            headlines,
            categories,
            test_size=self.test_size,
            random_state=SEED,
            stratify=categories,
        )
        del headlines, categories

        # Use headlines with at least 4 words/tokens for training
        x_train = x_train[x_train.apply(lambda x: len(x.split())) >= 4]
        y_train = y_train.loc[x_train.index]

        x_train = self.feature_extractor.fit_transform(x_train)

        self.model = CustomMLPSmall(
            in_neurons=x_train.shape[1], out_neurons=len(set(y_train))
        ).to(self.device)

        logging.info("Transforming validation set.")
        transformed_x_test = torch.tensor(
            self.feature_extractor.transform(x_test), device=self.device
        )

        logging.info(f"Beginning training of {self.NAME}")
        checkpoint = pd.Timestamp.now()

        for epoch in tqdm(range(n_epochs)):
            logging.info(f"\nEpoch: {epoch}\n")
            self.__train__(x_train, y_train, batch_size)

            with (torch.no_grad()):
                # Calling model directly instead of .predict() to
                # speed up the process by bypassing the transformations
                # and having them pre-computed in transformed_x_test
                _predictions = [
                    self.class_index_to_label[class_index.item()]
                    for class_index in self.model(transformed_x_test).argmax(1)
                ]

                validation_accuracy = utils.to_percentage(
                    (_predictions == y_test).mean()
                )
                logging.info(f"Validation accuracy: {validation_accuracy}")

        logging.info(
            f"Training took {(pd.Timestamp.now() - checkpoint).seconds}s"
        )

        self.model.eval()
        self.evaluate(x_test, y_test)

    def __train__(self, x_train, y_train, batch_size):
        dataloader = self.initialize_data_loader(x_train, y_train, batch_size)

        optimizer = Adam(self.model.parameters(), lr=0.0001)
        loss_criterion = self.initialize_loss_criterion(y_train)
        self.model.train()

        for step, (predictors, labels) in enumerate(dataloader):
            predictors = predictors.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            predictions = self.model(predictors)

            loss = loss_criterion(predictions, labels)
            loss = loss.mean()

            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                logging.info(f"Current Loss: {round(loss.item(), 4)}")

    def create_label_to_index_map(self, categories):
        _all_labels = set(categories)
        label_2_index = {
            label: index for index, label in enumerate(_all_labels)
        }
        index_2_label = {
            index: label for label, index in label_2_index.items()
        }
        self.label_to_class_index = label_2_index
        self.class_index_to_label = index_2_label

    def initialize_data_loader(self, x_train, y_train, batch_size=512):
        dataloader = DataLoader(
            CustomDataset(
                feature_matrix=x_train,
                labels=y_train,
                label_to_class_index=self.label_to_class_index,
            ),
            batch_size=batch_size,
        )
        return dataloader

    def initialize_loss_criterion(self, y_train):
        loss_criterion = CrossEntropyLoss(
            weight=torch.tensor(
                utils.get_inverse_class_distribution(y_train).values,
                dtype=torch.float32,
                device=self.device,
            ),
        )
        return loss_criterion

    def predict(self, headlines, **kwargs) -> Iterable[str]:
        model_input = torch.tensor(
            self.feature_extractor.transform(headlines),
            device=self.device,
            dtype=torch.float32,
        )
        predictions = self.model(model_input).argmax(axis=1)

        return [
            self.class_index_to_label[class_index.item()]
            for class_index in predictions
        ]

    def save(self, path=None, **kwargs) -> None:
        attributes = ["device", "label_to_class_index", "class_index_to_label"]
        to_save = {}

        for attribute in attributes:
            to_save[attribute] = getattr(self, attribute)

        super().save(path=path, **to_save)
