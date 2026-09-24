import pandas as pd

def display_results(row):
    return pd.Series([float(f"{row[metric]:.3g}") if metric not in ['regret', 'train_duration_per_epoch', 'total_train_duration'] else float(f"{row[metric]:.2f}") for metric in row.keys() if metric])