from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from news_classification.utils.utils import get_device


class CustomDataset(Dataset):
    def __init__(
        self,
        feature_matrix: Union[pd.DataFrame, np.ndarray],
        labels: pd.Series,
        label_to_class_index,
        device=get_device(),
    ):
        self.device = device

        self.label_to_class_index = label_to_class_index
        self.n_labels = len(label_to_class_index)

        self.feature_matrix = torch.tensor(feature_matrix, dtype=torch.float32)
        self.labels = torch.tensor(
            [label_to_class_index[label] for label in labels]
        )

    def __getitem__(self, index):
        feature_vector = self.feature_matrix[index, :]
        label = self.labels[index]

        return feature_vector, label

    def __len__(self):
        return len(self.labels)
