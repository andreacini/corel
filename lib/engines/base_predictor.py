import torch
from typing import Type, Mapping, Optional, Callable

from itertools import chain
from torchmetrics import Metric

from tsl.engines import Predictor


class BasePredictor(Predictor):
    def __init__(
        self,
        model_class: Type,
        model_kwargs: Mapping,
        optim_class: Type,
        optim_kwargs: Mapping,
        loss_fn: Optional[Callable] = None,
        readout_class: Optional[Type] = None,
        readout_kwargs: Optional[Mapping] = None,
        scale_target: bool = False,
        metrics: Optional[Mapping[str, Metric]] = None,
        scheduler_class: Optional[Type] = None,
        scheduler_kwargs: Optional[Mapping] = None,
    ):

        readout_kwargs = readout_kwargs or dict()
        if readout_class is not None:
            self.readout = readout_class(**readout_kwargs)
        else:
            self.readout = None

        super(BasePredictor, self).__init__(
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

    def forward(self, *args, readout_kwargs=None, **kwargs):
        """"""
        if self.filter_forward_kwargs:
            kwargs = self._filter_forward_kwargs(kwargs)
        out = self.model(*args, **kwargs)
        if self.readout is not None:
            out = self.readout(out, **readout_kwargs)
        return out

    def predict(self, *args, readout_kwargs=None, **kwargs):
        """"""
        return self.forward(*args, readout_kwargs, **kwargs)

    @property
    def trainable_parameters(self) -> int:
        """"""
        ps = super(BasePredictor, self).trainable_parameters
        if self.readout is not None:
            ps += sum(p.numel() for p in self.readout.parameters() if p.requires_grad)
        return ps

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """"""
        # Unpack batch
        x, y, mask, transform = self._unpack_batch(batch)

        # Make predictions
        y_hat = self.predict_batch(batch, preprocess=False, postprocess=True)

        output = dict(**y,
                      y_hat=y_hat,
                      batch_idx=torch.tensor([batch_idx], device=y_hat.device))
        if dataloader_idx is not None:
            output['dataloader_idx'] = torch.tensor([dataloader_idx], device=y_hat.device)
        if mask is not None:
            output['mask'] = mask
        return output

    def collate_prediction_outputs(self, outputs):
        """
        Collate the outputs of the :meth:`predict_step` method.

        Args:
            outputs: Collated outputs of the :meth:`predict_step` method.

        Returns:
            The collated outputs.
        """
        # handle single dataloader case
        if isinstance(outputs[0], dict):
            outputs = [outputs]
        # flatten list
        # outputs = [o for outs in outputs for o in outs] # Not very readable
        outputs = list(chain.from_iterable(outputs))

        return super(BasePredictor, self).collate_prediction_outputs(outputs)
