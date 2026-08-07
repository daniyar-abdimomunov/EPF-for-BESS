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
        https://www.google.com/search?atvm=2&source=hp&mstk=AUtExfABHoxXcWFQwHPsnAMR1CLfZpxdPTQYWqJ0yut7p3F905q9iXXcJa6U_zYwubXVh1Zz1fwZjJVdPqaKvil6AijXQBP7_MUl3hkItkEABYhv36AiKBzbw20sVTPRiR56zJq5ldsG0wWdeDjvIxEpSAq8pDIXgxHn3mkqcHEIJ7P_HQwvUBlKP9nSlPZ2uDa0ry8LodBxaNlGadg9YkF94g_g_aMNjvOacozpneZxE-uMlb0bnoWEZ72_eT5mTWjNdKqHyFwOpBUAyw&mtid=rno1asnPIceGxc8Pneq10AY&csuir=1&aep=26&q=Can+you+rewrite+the+Corr-f+dispersion+measures+proposed+by+Maciejowska+et+al.+(2026)+as+a+differentiable+loss+for+Torch+in+Python+for+model+training+that+takes+the+true+and+predicted+values+as+input+for+the+forward+method?$$%0A++%5Cbegin%7Beqnarray%7D%0A++Corr-f+%3D+%5Cfrac%7B1%7D%7BT%7D%5Csum%5Climits_%7Bt%3D1%7D%5E%7BT%7D%5Crho(P_t,+%5Chat%7BP_t%7D)%0A++%5Cend%7Beqnarray%7D%0A++$$&ved=0CAYQ2_wOahgKEwiQzM_3zPWVAxUAAAAAHQAAAAAQogI&lns_mode=cvst&udm=50
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
