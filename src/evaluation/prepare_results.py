import mlflow
import pandas as pd

# display select columns
params_columns = ['run_id', 'dataset', 'training_strategy', 'loss_configuration', 'seed']
metrics_columns = ['regret', 'train_duration_per_epoch', 'mae', 'rmse', 'cov_e', 'corr_f', ]
duration_columns = ['total_train_duration', 'total_epoch']

def prepare_results(experiment_name: str):
    # get all runs for experiment
    _runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        output_format = "pandas",
    )

    _runs = get_total_train_duration(mlflow.tracking.MlflowClient(), _runs)

    _runs = get_total_epoch(mlflow.tracking.MlflowClient(), _runs)

    # calculate train duration in minutes per epoch
    _runs['train_duration_per_epoch'] = _runs['total_train_duration'] / _runs['total_epoch'] / 60


    # relabel loss configurations
    loss_label = {
        'mae': 'MAE',
        'spo+': 'SPO+',
        'cov-e': 'Cov-e',
        'corr-f': 'Corr-f',
        'None': None
    }
    _runs['loss_configuration'] = _runs[['params.loss', 'params.penalty']].replace(loss_label).apply(
        lambda row: ' + '.join([l for l in row if l]), axis=1)
    _runs['loss_configuration'] = pd.Categorical(_runs['loss_configuration'],
                                                 categories=['MAE', 'MAE + SPO+', 'MAE + Cov-e', 'MAE + Corr-f'])
    _runs['params.training_mode'] = pd.Categorical(_runs['params.training_mode'],
                                                   categories=['train', 'pretrain', 'finetune'])

    # rename columns
    col_name_map = {
        'params.data_path': 'dataset',
        'params.seed': 'seed',
        'params.training_mode': 'training_strategy',
        'metrics.test_regret': 'regret',
        'metrics.test_mae': 'mae',
        'metrics.test_rmse': 'rmse',
        'metrics.test_cov-e': 'cov_e',
        'metrics.test_corr-f': 'corr_f',
    }

    # filters for selected datasets and seeds
    datasets_filter = _runs['params.data_path'].isin(['DE_LU.csv', 'DK1.csv', 'ES.csv'])

    _runs['params.seed'] = _runs['params.seed'].astype(int)
    seeds_filter = _runs['params.seed'].isin([123, 456, 7891011, 12131415, 643216842])



    # select runs
    runs = _runs.copy().rename(columns=col_name_map)
    runs = runs[(datasets_filter) & (seeds_filter)][params_columns + metrics_columns + duration_columns]
    runs = runs.sort_values(by=['dataset', 'training_strategy', 'loss_configuration', 'seed']).reset_index(drop=True)
    return runs

def get_total_train_duration(
        mlflow_client: mlflow.tracking.MlflowClient,
        runs
):
    _runs = runs.copy()
    # calculate total train duration from 'epoch_train_duration_sec' metric
    _runs['train_duration'] = _runs['run_id'].apply(lambda run_id: get_train_duration(mlflow_client, run_id))
    _runs['pre_train_duration'] = _runs.apply(
        lambda row:
        get_train_duration(
            mlflow_client,
            get_pretrain_run_id(row['experiment_id'], row['params.data_path'], row['params.seed'])
        ) if row['params.training_mode'] == 'finetune' else None,
        axis=1
    )
    _runs['total_train_duration'] = _runs['train_duration'] + _runs['pre_train_duration'].fillna(0)

    return _runs

def get_total_epoch(
        mlflow_client: mlflow.tracking.MlflowClient,
        runs
):
    _runs = runs.copy()
    # calculate total number of epochs for fine-tuned models
    _runs['pre_train_epoch'] = _runs.apply(
        lambda row:
        get_pretrain_epoch(
            mlflow_client,
            get_pretrain_run_id(row['experiment_id'], row['params.data_path'], row['params.seed'])
        ) if row['params.training_mode'] == 'finetune' else None,
        axis=1
    )
    _runs['finetune_start_epoch'] = _runs.apply(
        lambda row:
        get_finetune_start_epoch(
            mlflow_client,
            row['run_id']
        ) if row['params.training_mode'] == 'finetune' else None,
        axis=1
    )
    _runs['total_epoch'] = _runs['metrics.epoch'] + _runs['pre_train_epoch'].fillna(0) - _runs[
        'finetune_start_epoch'].fillna(0)
    _runs['total_epoch'] = _runs['total_epoch'].astype(int)
    return _runs

# define functions to calculate durations
def get_train_duration(
        client: mlflow.tracking.MlflowClient,
        run_id: str | None
):
    if run_id is None:
        return 0
    durations = client.get_metric_history(run_id=run_id, key='epoch_train_duration_sec')
    durations_values = [metric.value for metric in durations]
    return sum(durations_values)

def get_pretrain_run_id(
        experiment_id: str,
        data_path: str,
        seed: str | int,
):
    pt_run = mlflow.search_runs(
        experiment_ids = [experiment_id],
        filter_string = f"params.training_mode = 'pretrain'  AND params.data_path = '{data_path}' AND params.seed = '{seed}' "
    )
    pt_run_id = pt_run['run_id'].values[0] if len(pt_run) > 0 else None
    return pt_run_id

def get_pretrain_epoch(
        client: mlflow.tracking.MlflowClient,
        run_id: str | None
):
    if run_id is None:
        return 0
    run = client.get_run(run_id=run_id)
    epoch = run.to_dictionary()['data']['metrics']['epoch']
    return epoch

def get_finetune_start_epoch(
        client: mlflow.tracking.MlflowClient,
        run_id: str
):
    epochs = client.get_metric_history(run_id=run_id, key='epoch')
    epoch_values = [metric.value for metric in epochs]
    return sorted(epoch_values)[0]