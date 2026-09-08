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


# =============================================================================
# Utility & Loss Functions
# =============================================================================

def calculate_epistemic_uncertainty(output: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Extracts epistemic uncertainty (vacuity) from the ENN output dictionary."""
    return output["uncertainty"].squeeze(-1)


def evidential_loss(
    alpha: torch.Tensor,
    target: torch.Tensor,
    epoch: int,
    total_epochs: int,
    class_weights: torch.Tensor = None,
    lam: float = 0.05,
    anneal_epochs: int = 20,
) -> torch.Tensor:
    """
    Evidential Loss combining Expected Cross Entropy (NLL Type II) with a KL Divergence penalty.

    Why this specific formulation?
    Standard Cross Entropy forces the network to be overconfident.
    This loss function penalizes evidence placed on incorrect classes, pushing
    the Dirichlet distribution towards the corners of the simplex for correct predictions,
    while returning to a flat Dirichlet (high uncertainty) for out-of-distribution data.
    """
    K = alpha.shape[-1]
    S = alpha.sum(dim=-1, keepdim=True)

    # Convert integer targets to one-hot encoding
    target_oh = F.one_hot(target, num_classes=K).float()

    # 1. Expected Cross Entropy (Digamma formulation)
    # Replaces the standard log(p) to account for the Dirichlet distribution.
    loss_fit = target_oh * (torch.digamma(S) - torch.digamma(alpha))

    if class_weights is not None:
        loss_fit = loss_fit * class_weights

    loss_fit = loss_fit.sum(dim=-1).mean()

    # 2. KL Divergence Penalty (Hard Warmup)
    # The KL term penalizes evidence on incorrect classes.
    # Warmup is critical: we must allow the network to learn basic classification
    # features before enforcing strict uncertainty bounds, otherwise we risk gradient starvation.
    anneal_epochs = max(int(anneal_epochs), 0)
    if epoch <= anneal_epochs:
        kl_weight = 0.0
    else:
        # Progressive annealing up to ~40% of the training duration
        kl_weight = lam * min(
            1.0, (epoch - anneal_epochs) / max(total_epochs * 0.4, 1)
        )

    # alpha_tilde removes evidence from the true class to calculate the penalty
    # solely on the incorrect classes.
    alpha_tilde = target_oh + (1.0 - target_oh) * alpha
    ones = torch.ones_like(alpha_tilde)
    S_tilde = alpha_tilde.sum(dim=-1, keepdim=True)

    kl = (
            torch.lgamma(S_tilde)
            - torch.lgamma(torch.tensor(float(K), device=alpha.device))
            - torch.lgamma(alpha_tilde).sum(dim=-1, keepdim=True)
            + ((alpha_tilde - ones) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))).sum(dim=-1, keepdim=True)
    ).mean()

    return loss_fit + kl_weight * kl
