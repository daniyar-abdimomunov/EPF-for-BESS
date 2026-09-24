from numpy import argsort, divide, mean, ndarray, sqrt, sum, zeros_like

def corr_f(
        pred: ndarray,
        true: ndarray,
        reduction: str = 'mean'
):
    """
    Calculate Corr-f: The average row-wise Spearman correlation
    between true prices (P) and predicted prices (P_hat).

    Parameters:
        true (np.ndarray): Matrix of true values, shape [T, H, V]
        pred (np.ndarray): Matrix of predicted values, shape [T, H, V]
    """
    # Convert actual prices to ranks row-by-row (axis=1) for Spearman correlation.
    rank_P = argsort(argsort(true, axis=1), axis=1)
    rank_P_hat= argsort(argsort(pred, axis=1), axis=1)

    # Center the ranks by subtracting their row-wise means.
    P_centered = rank_P - mean(rank_P, axis=1, keepdims=True)
    P_hat_centered = rank_P_hat - mean(rank_P_hat, axis=1, keepdims=True)

    # Compute row-wise correlation on ranks.
    numerator = sum(P_centered * P_hat_centered, axis=1)
    denominator = sqrt(sum(P_centered ** 2, axis=1) * sum(P_hat_centered ** 2, axis=1))

    # Avoid division by zero if a target/prediction row is perfectly flat.
    rho_t = divide(numerator, denominator, out=zeros_like(numerator), where=denominator != 0)

    if reduction == 'none':
        return mean(rho_t, axis=1)
    elif reduction == 'mean':
        return mean(rho_t)