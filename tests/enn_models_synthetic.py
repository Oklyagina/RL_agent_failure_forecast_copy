"""Synthetic stand-in for src/enn_models.py used ONLY to validate the example
pipeline: same forward interface as the real EvidentialNetwork
(dict with "prob", "S", "uncertainty")."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialNetwork(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        evidence = F.softplus(self.net(x))          # [B, K] >= 0
        alpha = evidence + 1.0
        S = alpha.sum(dim=1)                        # [B]
        return {
            "prob": alpha / S.unsqueeze(1),
            "S": S,
            "uncertainty": self.num_classes / S,    # vacuity u = K/S
            "alpha": alpha,
        }
