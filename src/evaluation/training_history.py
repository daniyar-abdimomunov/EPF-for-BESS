from mlflow import search_runs
from mlflow.tracking import MlflowClient
import pandas as  pd
import seaborn as sns

import config

def plot_training_history(mlflow_client: MlflowClient, data_path: str, penalty: str):
    sns.set_theme(font='Times New Roman')
    EXPERIMENT_NAME = config.EXP_CONFIG['experiment_name']
    penalty_config =  {
        'cov-e': {
            'loss': 'CovELoss',
            'color': 'green',
        },
        'corr-f': {
            'loss': 'CorrFLoss',
            'color': 'red',
        },
    }
    loss = penalty_config[penalty]['loss']
    loss_names = ['loss', 'mae', loss]
    display_name = {
        'loss': f'Composite Loss: MAE + {loss}',
        'mae': 'Loss Component: MAE',
        loss: f'Loss Component: {loss}',
    }
    #
    _runs = search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string = f"params.training_mode = 'train' AND params.data_path = '{data_path}' AND params.penalty = '{penalty}' ",
    )
    run_ids = _runs['run_id'].values
    #
    all_metrics = pd.DataFrame()
    for run_id in run_ids:
        metrics_acc = list()
        for loss in loss_names:
            val_metric_history = mlflow_client.get_metric_history(run_id=run_id, key=f'val_{loss}')
            val_metric_values = pd.DataFrame(
                [(metric_item.step, 'val', display_name[loss], metric_item.value) for metric_item in val_metric_history],
                columns=['step', 'stage', 'loss', 'value'],
            )
            metrics_acc.append(val_metric_values)
            train_metric_history = mlflow_client.get_metric_history(run_id=run_id, key=f'train_{loss}_epoch')
            train_metric_values = pd.DataFrame(
                [(metric_item.step, 'train', display_name[loss], metric_item.value) for metric_item in train_metric_history],
                columns=['step', 'stage', 'loss', 'value'],
            )
            metrics_acc.append(train_metric_values)

        regret_history = mlflow_client.get_metric_history(run_id=run_id, key=f'val_regret')
        regret_values = pd.DataFrame(
            [(metric_item.step, 'val', 'Validation Metric: Regret', metric_item.value) for metric_item in regret_history],
            columns=['step', 'stage', 'loss', 'value'],
        )
        metrics_acc.append(regret_values)

        metrics = pd.concat(metrics_acc)
        #
        epoch_history = mlflow_client.get_metric_history(run_id=run_id, key='epoch')
        epochs = pd.DataFrame(
            [(metric.step, metric.value) for metric in epoch_history],
            columns=['step', 'epoch'],
        )
        metrics = pd.merge(metrics, epochs, on='step', how='left')
        metrics = metrics.dropna()
        metrics = metrics.drop_duplicates(keep='first')
        metrics['run_id'] = run_id

        all_metrics = pd.concat([all_metrics, metrics], ignore_index=True)
    ax = sns.relplot(
        kind='line',
        data=all_metrics,
        x='epoch',
        y='value',
        style='stage',
        col='loss',
        color=penalty_config[penalty]['color'],
        facet_kws={'sharey': False, 'sharex': True},
        height=3
    )
    ax.set_titles(template='{col_name}')

    return ax, all_metrics