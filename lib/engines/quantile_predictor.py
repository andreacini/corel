from typing import Type, Mapping, Optional

from einops import rearrange
from torchmetrics import Metric

from lib.engines.base_predictor import BasePredictor
from lib.metrics.torch_metrics.pinball_loss import MaskedMultiPinballLoss
from tsl.utils.python_utils import ensure_list

class QuantilePredictor(BasePredictor):
    def __init__(
        self,
        model_class: Type,
        model_kwargs: Mapping,
        optim_class: Type,
        optim_kwargs: Mapping,
        quantiles: list,
        scale_target: bool = False,
        metrics: Optional[Mapping[str, Metric]] = None,
        scheduler_class: Optional[Type] = None,
        scheduler_kwargs: Optional[Mapping] = None,
    ):
        self.quantiles = ensure_list(quantiles)
        self.n_quantiles = len(self.quantiles)
        self.n_targets = model_kwargs['output_size'] // self.n_quantiles

        loss_fn = MaskedMultiPinballLoss(qs=self.quantiles)

        super(QuantilePredictor, self).__init__(
            model_class=model_class,
            model_kwargs=model_kwargs,
            optim_class=optim_class,
            optim_kwargs=optim_kwargs,
            loss_fn=loss_fn,
            scale_target=scale_target,
            metrics=metrics,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
        )

    @staticmethod
    def _check_metric(metric, on_step=False):
        if not isinstance(metric, Metric):
            raise ValueError('Each metric must be an instance of Metric')
        metric = metric.clone()
        metric.reset()
        return metric

    def shared_step(self, batch, preprocess=False):
        y = y_loss = batch.y
        mask = batch.get('mask_target')

        # Compute predictions
        y_hat_loss = self.predict_batch(
            batch, preprocess=preprocess, postprocess=not self.scale_target
        )

        y_hat = y_hat_loss.detach()

        # Scale target and output, eventually
        if self.scale_target:
            y_loss = batch.transform['y'].transform(y)
            y_hat = batch.transform['y'].inverse_transform(y_hat)

        loss = self.loss_fn(y_hat_loss, y_loss, mask)

        return y_hat, y, loss, mask

    def training_step(self, batch, batch_idx):
        y_hat, y, loss, mask = self.shared_step(batch)

        self.train_metrics.update(y_hat, y, mask)
        self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
        self.log_loss('train', loss, batch_size=batch.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y, loss, mask = self.shared_step(batch)
        self.val_metrics.update(y_hat, y, mask)
        self.log_metrics(self.val_metrics, batch_size=batch.batch_size)
        self.log_loss('val', loss, batch_size=batch.batch_size)

        return loss

    def test_step(self, batch, batch_idx):
        y_hat, y, test_loss, mask = self.shared_step(batch)
        self.test_metrics.update(y_hat, y, mask)
        self.log_metrics(self.test_metrics, batch_size=batch.batch_size)
        self.log_loss('test', test_loss, batch_size=batch.batch_size)

        return test_loss


    def _unpack_batch(self, batch):
        """
        Unpack a batch into data and preprocessing dictionaries.

        :param batch: the batch
        :return: batch_data, batch_preprocessing
        """
        inputs, targets = batch.input, batch.target
        mask = batch.get('mask_target')
        transform = batch.get('transform')
        return inputs, targets, mask, transform

    def compute_metrics(self, batch, preprocess=False, postprocess=True):
        """"""
        raise NotImplementedError("compute_metrics not implemented for QuantilePredictor")


    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        out = super(QuantilePredictor, self).predict_step(batch, batch_idx, dataloader_idx)
        # reshape quantile to allow for stacking predictions
        out['y_hat'] = rearrange(out['y_hat'], 'q ... -> ... q')
        return out
