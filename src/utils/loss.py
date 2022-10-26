from typing import Tuple

import torch
from torch.nn import MSELoss, Module


class RegularizationLoss(Module):
    def __init__(self, strength: float, threshold: float):
        super().__init__()
        self.strength = strength
        self.threshold = threshold
        self.mse_loss = MSELoss()

    def forward(self, outputs: Tuple[torch.Tensor, torch.Tensor], targets: torch.Tensor):
        predictions, probabilities = outputs
        mse = self.mse_loss(predictions, targets)
        regularization = torch.sum(torch.relu(probabilities.sum(dim=0) - self.threshold))

        return mse + self.strength * regularization
