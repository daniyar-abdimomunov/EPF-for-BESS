import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback
from lightning.pytorch.loggers import MLFlowLogger
import os
import torch
from typing import Optional

from src.utils import TimeXerDataModule
from src.models import BESSTimeXer
from config import TrainingMode

NO_PENALTY_CONFIG = {
    'penalty': None,
    'penalty_lambda': None,
}

def execute_run(
        experiment_name: str,
        seed: int,
        data_config: dict,
        model_config: dict,
        battery_config: dict,
        training_mode: TrainingMode = TrainingMode.TRAIN,
        ckpt_dir: str = 'checkpoints',
        penalty_config: Optional[dict] = None,
        tracking_uri  = "sqlite:///mlflow.db",
        max_epochs: int = 10,
        do_test = True,
):
    L.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    if penalty_config is None:
        penalty_config = NO_PENALTY_CONFIG
    dataset_name  = data_config['data_path']
    run_name = _get_run_name(
        dataset_name=dataset_name,
        seed=seed,
        loss=model_config['loss'],
        training_mode=training_mode,
        **penalty_config,
    )
    print(f'Starting experiment: {run_name}...')

    # Setup Data
    data = TimeXerDataModule(**data_config | battery_config)
    
    # Setup Model
    model = _setup_model(
        ckpt_dir = ckpt_dir,
        dataset_name = dataset_name,
        seed = seed,
        model_config = model_config | data_config | battery_config | {'scaler': data.scaler},
        training_mode = training_mode,
        penalty_config = penalty_config
    )

    # Set up MLFlow Trainer
    hp = data_config | battery_config | model_config | penalty_config | {'seed': seed, 'training_mode': training_mode.value}
    trainer = _setup_trainer(experiment_name, run_name, tracking_uri, hp, ckpt_dir, max_epochs)

    # Train and test models using Trainer
    ckpt_path = os.path.join(ckpt_dir, run_name, 'best_model.ckpt')
    if os.path.exists(ckpt_path):
        """print(f"Checkpoint found at {ckpt_path}. Skipping training...")"""
        trainer.fit(model, datamodule=data, ckpt_path=ckpt_path)
        if do_test:
            print("Evaluating existing checkpoint...")
            trainer.test(model, datamodule=data, ckpt_path=ckpt_path)
    else:
        print('Training model...')
        if training_mode == TrainingMode.FINETUNE:
            ckpt_path = _get_finetune_checkpoint(ckpt_dir, dataset_name, seed, model_config['loss'])
        else:
            ckpt_path = None
        trainer.fit(model, datamodule=data, ckpt_path=ckpt_path)
        if do_test:
            trainer.test(model, datamodule=data, ckpt_path='best')
    return

def _get_run_name(
        dataset_name: str,
        seed: int,
        loss: str,
        training_mode: TrainingMode = TrainingMode.TRAIN,
        penalty: Optional[str] = None,
        penalty_lambda: Optional[float] = None,
) -> str:
    penalty, penalty_lambda, seed = [
        f'_{str}' if str is not None else '' for str  in [penalty, penalty_lambda, seed]
    ]
    run_name = f"{dataset_name}/{training_mode.value}/{loss}{penalty}{penalty_lambda}/seed{seed}"
    return run_name


def _setup_model(
        ckpt_dir: str,
        dataset_name: str,
        seed: int,
        model_config: dict,
        training_mode: TrainingMode,
        penalty_config: Optional[dict] = None,
):
    # Setup Model
    if training_mode == TrainingMode.FINETUNE:
        ckpt_path = _get_finetune_checkpoint(ckpt_dir, dataset_name, seed, model_config['loss'])

        print(f'Loading pre-trained model from checkpoint: {ckpt_path}')
        model = BESSTimeXer.load_from_checkpoint(ckpt_path, **model_config)
        model.set_phase(phase=training_mode, **penalty_config | {})
    else:
        model = BESSTimeXer(**model_config | penalty_config)
    return model


def _get_finetune_checkpoint(
        ckpt_dir: str,
        dataset_name: str,
        seed: int,
        loss: str,
):
    pretrained_model_run_name = _get_run_name(dataset_name, seed, loss, TrainingMode.PRETRAIN)

    # if finetune checkpoint exists, return finetune checkpoint path
    fine_tune_ckpt_path = os.path.join(ckpt_dir, pretrained_model_run_name, 'finetune_checkpoint.ckpt')
    if os.path.exists(fine_tune_ckpt_path):
        return fine_tune_ckpt_path

    # if pretrained model checkpoint exists, prepare checkpoint for fine-tuning
    try:
        pretrained_model_ckpt_path = os.path.join(ckpt_dir, pretrained_model_run_name, 'best_model.ckpt')
        checkpoint = torch.load(pretrained_model_ckpt_path)
        # Reset optimizer and callbacks.
        if "optimizer_states" in checkpoint:
            checkpoint["optimizer_states"] = []
        if "callbacks" in checkpoint:
            del checkpoint["callbacks"]
        torch.save(checkpoint, fine_tune_ckpt_path)
        return fine_tune_ckpt_path
    except Exception:
        print('No pretrained model checkpoint found. Please pre-train model first.')


def _setup_trainer(
        experiment_name: str,
        run_name: str,
        tracking_uri: str,
        hp: dict,
        ckpt_dir: str,
        max_epochs: int,
):
    mlflow_logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
        log_model=True,
    )
    mlflow_logger.log_hyperparams(hp)
    callbacks = _get_callbacks(dirpath=str(os.path.join(ckpt_dir, run_name)))
    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=mlflow_logger,
    )
    
    return trainer


def _get_callbacks(
        patience=3,
        dirpath="checkpoints",
        best_model_name="best_model"
) -> list[Callback]:

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min"
    )
    best_model_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        dirpath=dirpath,
        filename=best_model_name,
        save_top_k=1,
        mode="min"
    )
    
    return [early_stop, best_model_checkpoint]