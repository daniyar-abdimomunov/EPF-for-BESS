from argparse import Namespace
from lightning import LightningModule
from torch import argsort, cat, from_numpy, take_along_dim, Tensor, zeros_like
from torch.nn import L1Loss
from torch.optim import Adam
from typing import Optional

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
        if loss.lower() in  ('spo+', 'spoplus'):
            self.criterion = SPOPlus(optmodel=self.optModel)
        elif loss.lower() in ('corrf', 'corr-f'):
            self.criterion = CorrFLoss()
        elif loss.lower() in ('cove', 'cov-e'):
            self.criterion = CovELoss()
        else:
            self.criterion = L1Loss() # L1Loss == MAE
        return

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

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
        if isinstance(self.criterion, SPOPlus):
            # Re-scale prices to original prices for correct optimization.
            if self.scaler and self.inverse:
                outputs, batch_y = [self._rescale_predictions(tensor) for tensor in (outputs, batch_y)]
            preds_prices, true_prices = [tensor[:, :, self.f_dim] for tensor in [outputs, batch_y]]
            loss = self.criterion(preds_prices, true_prices)
        elif isinstance(self.criterion, CovELoss):
            hour_of_day = batch_y_mark[:, :, 0]
            outputs_aligned, batch_y_aligned = self._align_values(hour_of_day, outputs, batch_y)
            loss = self.criterion(outputs_aligned, batch_y_aligned)
        else:
            loss = self.criterion(outputs, batch_y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        # Do forward pass.
        outputs = self.forward(batch_x, batch_y, batch_x_mark, batch_y_mark)

        # Calculate validation metrics.
        outputs, batch_y, batch_y_mark = self._truncate_predictions(outputs, batch_y, batch_y_mark)
        if isinstance(self.criterion, SPOPlus) and self.scaler and self.inverse:
            if self.scaler and self.inverse:
                # Re-scale prices to original prices for correct optimization.
                outputs, batch_y = [self._rescale_predictions(tensor) for tensor in (outputs, batch_y)]
            preds_prices, true_prices = [tensor[:, :, self.f_dim] for tensor in [outputs, batch_y]]
            loss = self.criterion(preds_prices, true_prices).item()
        elif isinstance(self.criterion, CovELoss):
            hour_of_day = batch_y_mark[:, :, 0]
            outputs_aligned, batch_y_aligned = self._align_values(hour_of_day, outputs, batch_y)
            loss = self.criterion(outputs_aligned, batch_y_aligned).item()
        else:
            loss = self.criterion(outputs, batch_y).item()
        mae, rmse, corr_f, cov_e, regret = self._shared_eval_step(outputs, batch_y, batch_y_mark)

        # Log validation metrics.
        metrics = {
            'val_loss': loss,
            'val_mae': mae,
            'val_rmse': rmse,
            'val_corr_f': corr_f,
            'val_cov_e': cov_e,
            'val_regret': regret,
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return metrics

    def test_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        # Do forward pass.
        outputs = self.forward(batch_x, batch_y, batch_x_mark, batch_y_mark)

        # Calculate evaluation metrics.
        outputs, batch_y, batch_y_mark = self._truncate_predictions(outputs, batch_y, batch_y_mark)
        mae, rmse, corr_f, cov_e, regret = self._shared_eval_step(outputs, batch_y, batch_y_mark)

        # Log evaluation metrics.
        metrics = {
            'test_mae': mae,
            'test_rmse': rmse,
            'test_corr_f': corr_f,
            'test_cov_e': cov_e,
            'test_regret': regret,
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return metrics

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
    ) -> tuple[float, float, float, float, float]:
        # Convert tensors to numpy Arrays.
        preds, true = [
            tensor.cpu().detach().numpy()
            for tensor in (outputs, batch_y)
        ]

        # Calculate accuracy metrics.
        mae, rmse, corr_f = MAE(preds, true), RMSE(preds, true), corr_f_metric(preds, true)

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
        regret = regret_metric(preds_prices, true_prices, self.optModel)

        return mae, rmse, corr_f, cov_e, regret

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


