from torch import argsort, mean, sigmoid, sum, sqrt, Tensor
import torch.nn as nn

class CorrFLoss(nn.Module):
    """
    Trainable Spearman-based Corr-f loss function using a pairwise sigmoid soft-ranking approximation.
    Designed for tensor shapes [B, H, V], optimizing across the hourly profile axis (dim=1).
    Returns (1 - Soft_Spearman) as a minimization objective.
    """

    def __init__(self, temperature: float = 0.05, eps: float = 1e-8):
        super(CorrFLoss, self).__init__()
        self.temperature = temperature
        self.eps = eps

    def _compute_soft_rank(self, x: Tensor) -> Tensor:
        """
        Computes continuous soft ranks along the hour dimension (dim=1).
        Memory complexity per batch is lightweight: [B, V, H, H]
        """
        # Permute to isolate the profile dimension at the end: [B, V, H]
        x_perm = x.permute(0, 2, 1)

        # Expand dimensions to calculate cross-hour pairwise differences (x_i - x_j)
        x_col = x_perm.unsqueeze(-1)  # [B, V, H, 1]
        x_row = x_perm.unsqueeze(-2)  # [B, V, 1, H]
        diff_matrix = x_col - x_row

        # Soft relaxation using sigmoid
        soft_matrix = sigmoid(diff_matrix / self.temperature)

        # Sum comparisons along the row.
        # Add 0.5 to offset the diagonal self-comparison where sigmoid(0) = 0.5
        soft_ranks = sum(soft_matrix, dim=-1) + 0.5

        # Restore original dimension footprint: [B, H, V]
        return soft_ranks.permute(0, 2, 1)

    def forward(self, pred: Tensor, true: Tensor) -> Tensor:
        """
        TODO: add justification for modifications (soft rank for differentiation)
        Args:
            pred (Tensor): Continuous model outputs, shape [B, H, V]
            true (Tensor): Ground truth target constants, shape [B, H, V]
        """
        pred = pred.float()
        true = true.float()

        # 1. Targets do not need gradients: use your efficient hard double-argsort
        rank_P = argsort(argsort(true, dim=1), dim=1).float()

        # 2. Predictions need autograd tracking: use continuous soft ranks
        soft_rank_P_hat = self._compute_soft_rank(pred)

        # 3. Center the ranks by subtracting row-wise profile means
        P_centered = rank_P - mean(rank_P, dim=1, keepdim=True)
        P_hat_centered = soft_rank_P_hat - mean(soft_rank_P_hat, dim=1, keepdim=True)

        # 4. Compute Pearson correlation on these ranked spaces
        numerator = sum(P_centered * P_hat_centered, dim=1)
        denominator = sqrt(
            sum(P_centered ** 2, dim=1) * sum(P_hat_centered ** 2, dim=1)
        )

        # Smoothly avoid division by zero for flat rows
        rho_t = numerator / (denominator + self.eps)

        # 5. Average across all time intervals (B) and features (V)
        corr_f = mean(rho_t)

        # Return loss to minimize (0.0 = perfect alignment, 2.0 = total inversion)
        return 1.0 - corr_f
