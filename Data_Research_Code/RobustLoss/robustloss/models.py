# models.py
import torch
import torch.nn as nn

class LinearClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # logits

def build_model(n_features: int, num_classes: int) -> nn.Module:
    return LinearClassifier(n_features, num_classes)
