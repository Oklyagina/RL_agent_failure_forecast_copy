import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


# =============================================================================
# Neural Network Backbone Components
# =============================================================================

class ResidualBlock(nn.Module):
    """
    Standard Residual Block with Batch Normalization and Dropout.
    Helps mitigate vanishing gradients in deeper network configurations while
    providing regularization via Dropout.
    """

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Skip connection: f(x) + x
        return self.act(x + self.net(x))


class EvidentialNetwork(nn.Module):
    """
    Evidential Neural Network for evaluating policy familiarity in Grid2Op states.

    Architecture Design Rationale:
    Kept intentionally shallow (Embedding -> 2x ResBlocks -> Head) to prevent
    overfitting on small/noisy policy behavior-cloning datasets from Grid2Op.
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.05):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        # Initial feature projection
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Core representation learning
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout),
        ])

        # Evidential Head (Outputs raw logits, converted to Evidence later)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass converting raw logits into Dirichlet distribution parameters.

        Args:
            x (torch.Tensor): Normalized observation tensor (batch_size, input_dim).

        Returns:
            Dict containing:
                - alpha: Dirichlet parameters (evidence + 1)
                - evidence: Non-negative evidence from the network
                - S: Dirichlet strength (sum of alphas)
                - prob: Expected probabilities
                - uncertainty: Epistemic uncertainty (vacuity)
        """
        h = self.embedding(x)
        for block in self.res_blocks:
            h = block(h)

        logits = self.head(h)

        # Evidence must be non-negative. Softplus is used instead of ReLU
        # to ensure smooth gradients around zero.
        evidence = F.softplus(logits)

        # Dirichlet parameters (alpha) must be >= 1.
        # alpha = 1 implies zero evidence (total uncertainty/vacuity).
        alpha = evidence + 1.0

        # S is the total strength of the Dirichlet distribution
        sum_alpha = alpha.sum(dim=-1, keepdim=True)

        # Expected probability for each class
        prob = alpha / sum_alpha

        # Calculate epistemic uncertainty (vacuity).
        # High vacuity means the network has seen little/no data like this before.
        num_classes_tensor = torch.tensor(self.num_classes, dtype=x.dtype, device=x.device)
        vacuity = num_classes_tensor / sum_alpha

        return {
            "alpha": alpha,
            "evidence": evidence,
            "S": sum_alpha,
            "prob": prob,
            "uncertainty": vacuity,
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convenience method for inference."""
        self.eval()
        return self(x)
