import torch
from torch import nn as nn
from torch.nn import LeakyReLU


class CustomMLPSmall(nn.Module):
    def __init__(self, in_neurons: int, out_neurons: int, logits=True):
        super().__init__()
        self.logits = logits
        self.activation = LeakyReLU()

        self.network = nn.Sequential(
            nn.Linear(in_neurons, 1024),
            nn.BatchNorm1d(1024),
            self.activation,
            nn.Linear(1024, 512),
            self.activation,
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            self.activation,
            nn.Linear(256, out_neurons),
        )

    def forward(self, model_input: torch.Tensor):
        model_output = self.network(model_input)

        if not self.logits:
            model_output = self.soft_max(model_output)

        return model_output


class CustomMLPLarge(nn.Module):
    def __init__(self, in_neurons: int, out_neurons: int, logits=True):
        super().__init__()
        self.logits = logits
        self.activation = LeakyReLU()

        self.network = nn.Sequential(
            nn.Linear(in_neurons, 4096),
            nn.BatchNorm1d(4096),
            self.activation,
            nn.Linear(4096, 2048),
            self.activation,
            nn.Linear(2048, 1024),
            self.activation,
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            self.activation,
            nn.Linear(512, out_neurons),
        )

    def forward(self, model_input: torch.Tensor):
        model_output = self.network(model_input)

        if not self.logits:
            model_output = self.soft_max(model_output)

        return model_output
