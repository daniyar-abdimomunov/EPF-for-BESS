import pandas as pd
from src.evaluation.display_results import display_results

# display select columns
params_columns = ['run_id', 'dataset', 'training_strategy', 'loss_configuration', 'seed']
metrics_columns = ['regret', 'train_duration_per_epoch', 'mae', 'rmse', 'cov_e', 'corr_f', ]
duration_columns = ['total_train_duration', 'total_epoch']

def agg_results(runs, group_by: list[str]):
    runs_group_by = runs.groupby(group_by)
    runs_agg = pd.merge(
        runs_group_by[metrics_columns].agg('mean'), runs_group_by[duration_columns].agg('sum'),
        left_index=True,
        right_index=True
    )
    runs_agg['train_duration_per_epoch'] = runs_agg['total_train_duration'] / runs_agg['total_epoch'] / 60
    """
            pd.Series([float(
                f"{row[metric]:.3g}") if metric not in ['regret',
                                                        'train_duration_per_epoch'] else float(
                f"{row[metric]:.2f}") for metric in row.keys() if
                       metric]),"""

    runs_agg[metrics_columns] = runs_agg[metrics_columns].apply(lambda row: display_results(row),
        axis=1
        )
    runs_agg = runs_agg[metrics_columns].reset_index().dropna()
    return runs_agg