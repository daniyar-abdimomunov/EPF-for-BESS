import torch
from torch import diagonal, eye, linalg, log, matmul, Tensor
import torch.nn as nn

class CovELoss(nn.Module):
    """
    Differentiable PyTorch loss function based on the Cov-e metric.
    """
    def __init__(self, eps: float = 1e-8):
        """
        Initializes the CovELoss module.

        Args:
            eps (float): A small value added to the diagonal of the covariance matrix
                         for numerical stability to ensure it is positive definite.
        """
        super(CovELoss, self).__init__()
        self.eps = eps

    def forward(self, pred: Tensor, true: Tensor) -> Tensor:
        """
        Calculates the Cov-e loss.
        NOTE: to avoid numerical instability, ensure batch size [T] is sufficiently larger than pred_len *
        num_variables [H * V].
        TODO: add justification for modifications (ridge regularization, using Cholesky decomposition)
        ((link)[https://www.google.com/search?atvm=2&source=hp&mstk=AUtExfABHoxXcWFQwHPsnAMR1CLfZpxdPTQYWqJ0yut7p3F905q9iXXcJa6U_zYwubXVh1Zz1fwZjJVdPqaKvil6AijXQBP7_MUl3hkItkEABYhv36AiKBzbw20sVTPRiR56zJq5ldsG0wWdeDjvIxEpSAq8pDIXgxHn3mkqcHEIJ7P_HQwvUBlKP9nSlPZ2uDa0ry8LodBxaNlGadg9YkF94g_g_aMNjvOacozpneZxE-uMlb0bnoWEZ72_eT5mTWjNdKqHyFwOpBUAyw&mtid=rno1asnPIceGxc8Pneq10AY&csuir=1&aep=26&q=Can+you+rewrite+the+Corr-f+dispersion+measures+proposed+by+Maciejowska+et+al.+(2026)+as+a+differentiable+loss+for+Torch+in+Python+for+model+training+that+takes+the+true+and+predicted+values+as+input+for+the+forward+method?$$%0A++%5Cbegin%7Beqnarray%7D%0A++Corr-f+%3D+%5Cfrac%7B1%7D%7BT%7D%5Csum%5Climits_%7Bt%3D1%7D%5E%7BT%7D%5Crho(P_t,+%5Chat%7BP_t%7D)%0A++%5Cend%7Beqnarray%7D%0A++$$&ved=0CAYQ2_wOahgKEwiQzM_3zPWVAxUAAAAAHQAAAAAQogI&lns_mode=cvst&udm=50])

        Inputs:
        - pred: torch.Tensor of shape [T, H, V] (Days, Hours)
        - true: torch.Tensor of shape [T, H, V] (Days, Hours)

        Returns:
        - loss: scalar torch.Tensor, the log-determinant of the covariance matrix.
        """
        # Calculate forecast errors for each day and hour
        e = pred - true  # Shape: [T, H]
        T = e.shape[0]

        # Reshape errors to combine H (Hours) and V (Variables) into a single dimension
        e_reshaped = e.reshape(T, -1)  # Shape: [T, H * V]

        # Compute the H x H variance-covariance matrix (Sigma hat)
        sigma_hat = (1/T) * matmul(e_reshaped.T, e_reshaped)   # Shape: [H * V, H * V]

        # Add ridge regularization to the diagonal to prevent rank deficiency
        identity = eye(sigma_hat.shape[0], device=e.device, dtype=sigma_hat.dtype)
        sigma_hat = sigma_hat + (identity * self.eps)

        # Stable Log-Determinant via Cholesky Decomposition:
        # If A = L @ L^T, then log|A| = 2 * sum(log(diag(L)))
        try:
            L = linalg.cholesky(sigma_hat)
            # Extract the diagonal elements of the lower triangular matrix L
            L_diag = diagonal(L, dim1=-2, dim2=-1)
            # Compute 2 * sum(log(diag(L)))
            loss = 2.0 * sum(log(L_diag))
        except torch._C._LinAlgError:
            # Fallback to slogdet if Cholesky fails due to extreme numerical edge cases
            _, loss = linalg.slogdet(sigma_hat)

        return loss
