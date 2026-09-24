from numpy import dot, linalg, ndarray

def cov_e(
        pred: ndarray,
        true: ndarray,
):
    """
    Calculate the Cov-e metric based on the log-determinant of the
    error variance-covariance matrix, supporting multivariate predictions.

    Parameters:
        true (np.ndarray): Matrix of true values, shape [T, H, V]
        pred (np.ndarray): Matrix of predicted values, shape [T, H, V]
    """
    # Calculate forecast errors for each day, hour, and variable.
    e = pred - true  # Shape: [T, H, V]
    T = e.shape[0]

    # Reshape errors to combine H (Hours) and V (Variables) into a single dimension.
    e_reshaped = e.reshape(T, -1)  # Shape: [T, H * V]

    # Compute the variance-covariance matrix (Sigma hat)
    sigma_hat = (1 / T) * dot(e_reshaped.T, e_reshaped)  # Shape: [H * V, H * V]

    _, cov_e = linalg.slogdet(sigma_hat)

    return cov_e