from argparse import Namespace
from lightning import LightningModule
from torch import cat, Tensor, zeros_like
from torch.nn import L1Loss
from torch.optim import Adam

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
        outputs, batch_y = self._truncate_predictions(outputs, batch_y)
        loss = self.criterion(outputs, batch_y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        # Do forward pass.
        outputs = self.forward(batch_x, batch_y, batch_x_mark, batch_y_mark)

        # Calculate validation metrics.
        outputs, batch_y = self._truncate_predictions(outputs, batch_y)
        loss = self.criterion(outputs, batch_y).item()
        mae, rmse = self._shared_eval_step(outputs, batch_y)

        # Log validation metrics.
        metrics = {
            'val_loss': loss,
            'val_mae': mae,
            'val_rmse': rmse,
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return metrics

    def test_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        # Do forward pass.
        outputs = self.forward(batch_x, batch_y, batch_x_mark, batch_y_mark)

        # Calculate evaluation metrics.
        outputs, batch_y = self._truncate_predictions(outputs, batch_y)
        mae, rmse = self._shared_eval_step(outputs, batch_y)

        # Log evaluation metrics.
        metrics = {
            'test_mae': mae,
            'test_rmse': rmse,
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
    ) -> tuple[Tensor, Tensor]:
        # Truncate output to only include predictions and
        # depending on whether predictions are univariate or multivariate.
        outputs = outputs[:, -self.pred_len:, self.f_dim:]
        batch_y = batch_y[:, -self.pred_len:, self.f_dim:]

        return outputs, batch_y

    def _shared_eval_step(
            self,
            outputs: Tensor,
            batch_y: Tensor,
    ) -> tuple[float, float]:
        # Convert tensors to numpy Arrays.
        preds, true = [
            tensor.cpu().detach().numpy()
            for tensor in (outputs, batch_y)
        ]

        # Calculate accuracy metrics.
        mae, rmse = MAE(preds, true), RMSE(preds, true)

        return mae, rmse
