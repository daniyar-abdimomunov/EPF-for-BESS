from argparse import Namespace
from lightning import LightningModule
from numpy import mean
from scipy.stats import spearmanr
from torch import argsort, cat, from_numpy, take_along_dim, Tensor, zeros_like
from torch.nn import L1Loss, Module
from torch.optim import Adam
from typing import Callable, Optional

from src.metrics import (
    corr_f as corr_f_metric,
    cov_e as cov_e_metric,
    regret as regret_metric
)
from src.losses import CorrFLoss, CovELoss, SPOPlus
from src.utils import BESSSchedulingOptModel
from timexer.models.TimeXer import Model
from timexer.utils.metrics import MAE, RMSE


class BESSTimeXer(LightningModule):
    def __init__(
            self,
            features: str = 'MS',
            pred_len: int = 24 * 1,
            label_len: int = 48 * 2,
            inverse: bool = False,
            output_attention: bool = False,
            loss: str = 'mae',
            penalty: Optional[str] = None,
            penalty_lambda: Optional[float] = None,
            scaler = None,
            learning_rate: float = 1e-4,
            **kwargs,
    ):
        """
        LightningModule wrapper for TimeXer model.
        """
        super().__init__()
        self.features = features
        self.f_dim = -1 if self.features == 'MS' else 0
        self.pred_len = pred_len
        self.label_len = label_len
        self.inverse = inverse
        self.output_attention = output_attention

        self.model = Model(configs = Namespace(
            **kwargs,
            features=features,
            pred_len=pred_len,
            output_attention=output_attention,
        ))
        self.scaler = scaler
        self.learning_rate = learning_rate
        self.optModel = BESSSchedulingOptModel(num_timesteps=pred_len)
        self.loss_name = loss
        self.penalty_name = penalty
        self.penalty_lambda = penalty_lambda
        self.loss_fn, self.criterion, self.penalty = self._construct_loss_fn(self.loss_name, self.penalty_name, self.penalty_lambda)
        self.current_phase = 'pretrain'

        self._eval_sample_metrics = {
            'mae': [],
            'rmse': [],
            'corr-f': [],
            'regret': [],
        }
        self._eval_batch_metrics = {
            'mae': [],
            'rmse': [],
            'corr-f': [],
            'cov-e': [],
            'regret': [],
        }
        return

    def _construct_loss_fn(self, loss_name: str, penalty_name: Optional[str], penalty_lambda: Optional[float]):
        criterion, _criterion_fn = self._get_criterion(loss_name)
        def loss_fn(
                outputs: Tensor,
                batch_y: Tensor,
                flag: str = 'train',
                *args: Tensor,
                **kwargs: Tensor,
        ) -> dict[str, Tensor]:
            loss_value = _criterion_fn(outputs, batch_y, *args, **kwargs)
            loss_value = loss_value.item() if flag == 'val' else loss_value
            return {
                f'{flag}_loss': loss_value,
            }

        if not penalty_name or not penalty_lambda:
            return loss_fn, criterion, None

        penalty_name, _penalty_fn = self._get_criterion(penalty_name)
        def composite_loss_fn(
                outputs: Tensor,
                batch_y: Tensor,
                flag: str = 'train',
                *args: Tensor,
                **kwargs: Tensor,
        ) -> dict[str, Tensor]:
            base_loss_value = _criterion_fn(outputs, batch_y, *args, **kwargs)
            penalty_value = penalty_lambda * _penalty_fn(outputs, batch_y, *args, **kwargs)
            loss_value = base_loss_value + penalty_value
            if flag == 'val':
                loss_value, base_loss_value, penalty_value = [value.item() for value in (loss_value, base_loss_value, penalty_value)]
            return {
                f'{flag}_loss': loss_value,
                f'{flag}_{loss_name}': base_loss_value,
                f'{flag}_{penalty_name}': penalty_value
            }

        return composite_loss_fn, criterion, penalty_name

    def _get_criterion(self, name: str) -> (Module, Callable[[Tensor, Tensor, Optional[Tensor]], Tensor]):
        _prepare_criterion_inputs_fn = self._pass_inputs
        name_safe = name.lower().replace('-', '')
        match name_safe:
            case 'corrf':
                criterion = CorrFLoss()
            case 'cove':
                criterion = CovELoss()
                _prepare_criterion_inputs_fn = self._prepare_cove_inputs
            case 'spo+' | 'spoplus':
                criterion = SPOPlus(optmodel=self.optModel)
                _prepare_criterion_inputs_fn = self._prepare_spoplus_inputs
            case 'mae':
                criterion = L1Loss()
            case _:
                raise ValueError(f'Criterion: {name} not supported.')

        _criterion_fn = lambda outputs, batch_y, *args, **kwargs: (
            criterion(*_prepare_criterion_inputs_fn(outputs, batch_y, *args, **kwargs))
        )

        return criterion, _criterion_fn

    def _pass_inputs(
            self,
            outputs: Tensor,
            batch_y: Tensor,
            *args: Tensor,
            **kwargs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return outputs, batch_y

    def _prepare_cove_inputs(
            self,
            outputs: Tensor,
            batch_y: Tensor,
            batch_y_mark: Tensor,
            *args: Tensor,
            **kwargs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        hour_of_day = batch_y_mark[:, :, 0]
        outputs_aligned, batch_y_aligned = self._align_values(hour_of_day, outputs, batch_y)
        return outputs_aligned, batch_y_aligned

    def _prepare_spoplus_inputs(
            self,
            outputs: Tensor,
            batch_y: Tensor,
            *args: Tensor,
            **kwargs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # Re-scale prices to original prices for correct optimization.
        if self.scaler and self.inverse:
            outputs, batch_y = [self._rescale_predictions(tensor) for tensor in (outputs, batch_y)]
        preds_prices, true_prices = [tensor[:, :, self.f_dim] for tensor in [outputs, batch_y]]
        return preds_prices, true_prices

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        return Adam(trainable_params, lr=self.learning_rate, weight_decay=1e-3)

    def set_phase(
            self,
            phase: str,
            lr: Optional[float] = 1e-5,
            loss: Optional[str] = None,
            penalty: Optional[str] = None,
            penalty_lambda: Optional[float] = None
    ):
        self.current_phase = phase
        self.learning_rate = lr
        self.loss_name = loss if loss is not None else self.loss_name
        self.penalty_name = penalty if penalty is not None else self.penalty_name
        self.penalty_lambda = penalty_lambda if penalty_lambda is not None else self.penalty_lambda
        self.loss_fn, self.criterion, self.penalty = self._construct_loss_fn(self.loss_name, self.penalty_name,
                                                                             self.penalty_lambda)
        if phase == 'finetune':
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.head.parameters():
                param.requires_grad = True
        return

    def forward(
            self,
            batch_x: Tensor,
            batch_y: Tensor,
            batch_x_mark: Tensor,
            batch_y_mark: Tensor,
    ) -> Tensor:
        # Prepare decoder input.
        dec_inp = self._decoder_input(batch_y)

        # Calculate model predictions.
        if self.output_attention:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        return outputs

    def training_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        # Do forward pass.
        outputs = self.forward(batch_x, batch_y, batch_x_mark, batch_y_mark)

        # Calculate loss.
        outputs, batch_y, batch_y_mark = self._truncate_predictions(outputs, batch_y, batch_y_mark)
        loss_items = self.loss_fn(outputs, batch_y, batch_y_mark=batch_y_mark, flag='train')
        self.log_dict(loss_items, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True)
        return loss_items['train_loss']

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        # Do forward pass.
        outputs = self.forward(batch_x, batch_y, batch_x_mark, batch_y_mark)

        # Calculate validation metrics.
        outputs, batch_y, batch_y_mark = self._truncate_predictions(outputs, batch_y, batch_y_mark)
        loss_items = self.loss_fn(outputs, batch_y, batch_y_mark=batch_y_mark, flag='val')
        metrics = self._shared_eval_step(outputs, batch_y, batch_y_mark, flag='val')

        # Log validation metrics.
        metrics.update(loss_items)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return metrics

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end(flag='val')
        return

    def test_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        # Do forward pass.
        outputs = self.forward(batch_x, batch_y, batch_x_mark, batch_y_mark)

        # Calculate evaluation metrics.
        outputs, batch_y, batch_y_mark = self._truncate_predictions(outputs, batch_y, batch_y_mark)
        metrics = self._shared_eval_step(outputs, batch_y, batch_y_mark, flag = 'test')

        # Log evaluation metrics.
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return metrics

    def on_test_epoch_end(self):
        self._on_eval_epoch_end(flag='test')
        return

    def predict_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch

        dec_inp = self._decoder_input(batch_y)
        dec_out = self.model.forecast(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.scaler and self.inverse:
            dec_out = self._rescale_predictions(dec_out)

        dec_out = dec_out.cpu().detach().numpy()
        return dec_out

    def _decoder_input(self, batch_y: Tensor) -> Tensor:
        dec_inp = zeros_like(batch_y[:, -self.pred_len:, :])
        dec_inp = cat([batch_y[:, :self.label_len, :], dec_inp], dim=1)
        return dec_inp

    def _truncate_predictions(
            self,
            outputs: Tensor,
            batch_y: Tensor,
            batch_y_mark: Optional[Tensor] = None,
    ) -> tuple[Tensor, ...]:
        # Truncate output to only include predictions and
        # depending on whether predictions are univariate or multivariate.
        outputs = outputs[:, -self.pred_len:, self.f_dim:]
        batch_y = batch_y[:, -self.pred_len:, self.f_dim:]

        # Truncate time encodings to only include encodings for predicted period_truncate_predictions
        batch_y_mark = batch_y_mark[:, -self.pred_len:, :] if batch_y_mark is not None else None

        return tuple(tensor for tensor in [outputs, batch_y, batch_y_mark] if tensor is not None)

    def _shared_eval_step(
            self,
            outputs: Tensor,
            batch_y: Tensor,
            batch_y_mark: Tensor,
            flag: str = 'val',
    ) -> dict[str, float]:
        # Convert tensors to numpy Arrays.
        preds, true = [
            tensor.cpu().detach().numpy()
            for tensor in (outputs, batch_y)
        ]

        # Calculate accuracy metrics.
        mae, rmse, corr_f = MAE(preds, true, reduction='none'), RMSE(preds, true, reduction='none'), corr_f_metric(preds, true, reduction='none')

        # Re-scale true and predicted values to original scale for further metrics.
        if self.scaler and self.inverse:
            outputs, batch_y = [self._rescale_predictions(tensor) for tensor in (outputs, batch_y)]

        # Align true and predicted values by hour of day, to calculate cov-e
        hour_of_day = batch_y_mark[:, :, 0]
        outputs_aligned, batch_y_aligned = self._align_values(hour_of_day, outputs, batch_y)
        preds_aligned, true_aligned = [ tensor.cpu().detach().numpy() for tensor in (outputs_aligned, batch_y_aligned) ]
        cov_e = cov_e_metric(preds_aligned, true_aligned)

        # Extract only prices from tensors and convert to numpy Arrays.
        preds_prices, true_prices = [ tensor[:, :, -1].cpu().detach().numpy() for tensor in (outputs, batch_y) ]
        regret = regret_metric(preds_prices, true_prices, self.optModel, reduction='none')

        for key, metric in zip(self._eval_sample_metrics, [mae, rmse, corr_f, regret]):
            self._eval_sample_metrics[key].extend(metric)

        metrics = dict()
        for key, metric in zip(self._eval_batch_metrics, [mae, rmse, corr_f, cov_e, regret]):
            batch_metric = mean(metric)
            self._eval_batch_metrics[key].append(batch_metric)
            metrics[f'{flag}_{key}'] = batch_metric

        return metrics

    def _on_eval_epoch_end(self, flag = 'val'):
        metric_corrs = dict()
        sample_corrs = self._calculate_metric_correlations(level='sample')
        metric_corrs.update(sample_corrs)

        batch_corrs = self._calculate_metric_correlations(level='batch')
        metric_corrs.update(batch_corrs)

        self._clear_metrics_acc()

        metric_corrs = dict((f"{flag}_{k}", v) for k, v in metric_corrs.items())
        self.log_dict(metric_corrs, on_step=False, on_epoch=True, prog_bar=True)
        return

    def _calculate_metric_correlations(self, level: str = 'batch'):
        metrics_acc = self._eval_batch_metrics.copy() if level == 'batch' else self._eval_sample_metrics.copy()
        regrets_acc = metrics_acc.pop('regret')

        corrs = dict()
        for key in metrics_acc.keys():
            print(len(metrics_acc[key]))
            corrs[f'{key}<>regret_{level}_corr'] = spearmanr(metrics_acc[key], regrets_acc).statistic

        return corrs

    def _clear_metrics_acc(self):
        for key in self._eval_sample_metrics.keys():
            self._eval_sample_metrics[key].clear()
        for key in self._eval_batch_metrics.keys():
            self._eval_batch_metrics[key].clear()
        return

    def _rescale_predictions(self, tensor: Tensor) -> Tensor:
        mean = from_numpy(self.scaler.mean_[self.f_dim:]).to(tensor.device, dtype=tensor.dtype)
        scale = from_numpy(self.scaler.scale_[self.f_dim:]).to(tensor.device, dtype=tensor.dtype)

        tensor = tensor * scale + mean

        return tensor

    def _align_values(
            self,
            index: Tensor,
            *tensors: Tensor,
    ) -> tuple[Tensor, ...]:
        # Get indices based on e.g. hour of day.
        indices = argsort(index, dim=1)

        # Expand indices if tensors are 3-dimensional.
        if all([tensor.dim() == 3 for tensor in tensors]):
            indices = indices[:, :, None]

        # Sort tensors by indices.
        tensors = tuple(take_along_dim(tensor.cpu(), indices.cpu(), dim=1) for tensor in tensors)
        return tensors


