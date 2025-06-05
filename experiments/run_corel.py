import os

import numpy as np
import pandas as pd
import torch
import yaml

from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger

from lib.nn.decoder.multiquantile_readout import MultiQuantileDecoder
from lib.nn.encoder_decoder_model import EncoderDecoderModel
from lib.nn.encoders.corel_encoder import CoRelEncoder
from lib.nn.encoders.rnn_encoder import RNNEncoder

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from lib.datasets.gpvar import GPVARDataset
from lib.datasets.air_quality import AirQuality
from lib.engines.quantile_predictor import QuantilePredictor
from lib.metrics.torch_metrics.coverage import MaskedCoverage, MaskedDeltaCoverage, MaskedPIWidth
from lib.metrics.torch_metrics.pinball_loss import MaskedMultiPinballLoss
from lib.metrics.torch_metrics.winkler import MaskedWinklerScore
from lib.utils.data_utils import parse_and_filter_indices, find_close
from tsl import logger

from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule, BatchMap, BatchMapItem
from tsl.data.datamodule.splitters import FixedIndicesSplitter
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import MetrLA
from tsl.experiment import Experiment

from lib.metrics.torch_metrics.wrappers import MaskedMetricWrapper

def get_encoder_class(encoder_str):
    # Basic models  #####################################################
    if encoder_str == 'rnn':
        encoder = RNNEncoder
    elif encoder_str == 'corel':
        encoder = CoRelEncoder
    else:
        raise NotImplementedError(f'Model "{encoder_str}" not available.')
    return encoder


def get_decoder_class(decoder_str):
    # Basic models  #####################################################
    if decoder_str == 'multiquantile':
        readout = MultiQuantileDecoder
    else:
        raise NotImplementedError(f'Model "{decoder_str}" not available.')
    return readout


def get_dataset(dataset_cfg):
    name = dataset_cfg["name"]
    if name == 'la':
        dataset = MetrLA()
    elif name == 'air':
        dataset = AirQuality()
    elif name == 'gpvar':
        dataset = GPVARDataset(**dataset_cfg["hparams"], p_max=0)
    else:
        raise ValueError(f"Dataset {name} not available.")
    return dataset


TARGET_QUANTILES = np.round(np.arange(0.025, 1, 0.025), 3).tolist()


def run_experiment(cfg: DictConfig):
    ########################################
    # data module                          #
    ########################################


    local_dir = cfg.src_dir

    residuals_input: pd.DataFrame = pd.read_hdf(os.path.join(local_dir, "residuals.h5"), key='input')
    residuals_target: pd.DataFrame = pd.read_hdf(os.path.join(local_dir, "residuals.h5"), key='target')
    with open(os.path.join(local_dir, "config.yaml"), 'r') as fp:
        src_config = yaml.load(fp, Loader=yaml.FullLoader)

    assert cfg.dataset.name == src_config["dataset"]["name"]
    dataset = get_dataset(src_config["dataset"])

    if cfg.dataset.name in {'gpvar'}:
        ds_index = pd.Index(dataset.index)
    else:
        ds_index = dataset.index

    try:
        mask_target = pd.read_hdf(os.path.join(local_dir, "residuals.h5"), key='target_mask')
        mask_target = mask_target.reindex(index=dataset.index)
    except KeyError:
        mask_target = None

    indices = np.load(os.path.join(local_dir, "indices.npz"))

    covariates = dict(
        residuals_input=(residuals_input.reindex(index=ds_index), 't n f'),
        residuals_target=(residuals_target.reindex(index=ds_index), 't n f'),
    )
    if cfg.get('add_exogenous'):
        # encode time of the day and use it as exogenous variable
        assert src_config["dataset"]["name"] not in {'gpvar', 'toy', 'mso'}
        day_sin_cos = dataset.datetime_encoded('day').values
        weekdays = dataset.datetime_onehot('weekday').values
        covariates.update(u=np.concatenate([day_sin_cos, weekdays], axis=-1))

    # use residuals as regression targets
    target_map = BatchMap()
    target_map['y'] = BatchMapItem('residuals_target',
                                   synch_mode='horizon',
                                   pattern='t n f',
                                   preprocess=False)

    input_map = BatchMap()

    if mask_target is not None:
        input_map['mask_target'] = BatchMapItem(['mask_target'],
                                                synch_mode='horizon',
                                                pattern='t n f')
        covariates.update(mask_target=mask_target.astype('bool'))

    if 'u' in covariates:
        input_map['u'] = BatchMapItem('u',
                                      synch_mode='window',
                                      pattern='t f')

    inputs_ = ['residuals_input']
    model_input_size = dataset.n_channels * src_config["horizon"]

    if cfg.get("target_as_input", True):
        inputs_.append('target')
        model_input_size += dataset.n_channels

    input_map['x'] = BatchMapItem(inputs_,
                                  synch_mode='window',
                                  pattern='t n f')

    torch_dataset = SpatioTemporalDataset(index=ds_index,
                                          target=dataset.dataframe(),  # original target time series
                                          mask=dataset.mask,
                                          covariates=covariates,
                                          window=cfg.window,
                                          stride=src_config["stride"],
                                          target_map=target_map,
                                          input_map=input_map,
                                          delay=src_config.get("delay", 0),
                                          horizon=1)

    calib_indices, test_indices = parse_and_filter_indices(torch_dataset, indices)

    val_len = int(cfg.val_len * len(calib_indices))
    calib_indices, val_indices = calib_indices[:-val_len - torch_dataset.samples_offset], calib_indices[-val_len:]

    calib_splitter = FixedIndicesSplitter(
        train_idxs=calib_indices,
        val_idxs=val_indices,
        test_idxs=test_indices,
    )

    scale_axis = (0,) if cfg.get('scale_axis') == 'node' else (0, 1)
    transform = {
        'target': StandardScaler(axis=scale_axis),
        'residuals_target': StandardScaler(axis=scale_axis),
        'residuals_input': StandardScaler(axis=scale_axis),
    }
    # transform = {}

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=transform,
        splitter=calib_splitter,
        batch_size=cfg.batch_size,
        workers=cfg.workers
    )
    dm.setup()

    ########################################
    # Create model                         #
    ########################################

    alphas = sorted(cfg.alphas)
    assert alphas[-1] < .5

    target_qs = TARGET_QUANTILES

    d_exog = torch_dataset.input_map.u.shape[-1] if 'u' in torch_dataset else 0

    model_class = EncoderDecoderModel

    ## Encoder init
    encoder_cls = get_encoder_class(cfg.model.encoder.name)
    encoder_kwargs = dict(window=torch_dataset.window,
                          n_instances=torch_dataset.n_nodes,
                          )
    encoder_cls.filter_init_args_(encoder_kwargs)
    encoder_kwargs.update(cfg.model.encoder.hparams)

    ## Decoder init
    decoder_cls = get_decoder_class(cfg.model.decoder.name)

    decoder_kwargs = dict(quantiles=target_qs)
    decoder_cls.filter_init_args_(decoder_kwargs)
    decoder_kwargs.update(cfg.model.decoder.hparams)

    model_kwargs = dict(encoder_class=encoder_cls,
                        encoder_kwargs=encoder_kwargs,
                        decoder_class=decoder_cls,
                        decoder_kwargs=decoder_kwargs,
                        input_size=model_input_size,
                        exog_size=d_exog,
                        output_size=torch_dataset.n_channels * src_config["horizon"],
                        horizon=torch_dataset.horizon)

    model_class.filter_model_args_(model_kwargs)
    model_kwargs.update(cfg.model.hparams)

    ########################################
    # Metrics.                             #
    ########################################

    def get_metric_at_alpha(base_metric, target_alpha):
        idx_low = find_close(target_alpha / 2, target_qs)
        idx_high = find_close(1 - target_alpha / 2, target_qs)

        def preprocessing_fn(y_hat):
            return torch.stack((y_hat[idx_low], y_hat[idx_high]))

        return MaskedMetricWrapper(metric=base_metric,
                                   input_preprocessing=preprocessing_fn)

    log_metrics = {
        'pinball': MaskedMultiPinballLoss(qs=target_qs),
        }

    for a in alphas:
        log_metrics[f'coverage_at_{int((1 - a) * 100)}'] = get_metric_at_alpha(MaskedCoverage(), a)
        log_metrics[f'delta_cov_at_{int((1 - a) * 100)}'] = get_metric_at_alpha(MaskedDeltaCoverage(alpha=a), a)
        log_metrics[f'pi_width_at_{int((1 - a) * 100)}'] = get_metric_at_alpha(MaskedPIWidth(), a)
        log_metrics[f'winkler_at_{int((1 - a) * 100)}'] = get_metric_at_alpha(MaskedWinklerScore(alpha=a), a)

    log_metrics = {k.replace(".", "-"): v for k, v in log_metrics.items()}

    if cfg.get('lr_scheduler') is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler,
                                  cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None

    predictor = QuantilePredictor(
        model_class=model_class,
        model_kwargs=model_kwargs,
        optim_class=getattr(torch.optim, cfg.optimizer.name),
        optim_kwargs=dict(cfg.optimizer.hparams),
        quantiles=target_qs,
        metrics=log_metrics,
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        scale_target=cfg.scale_target,
    )

    ########################################
    # logging options                      #
    ########################################

    run_args = exp.get_config_dict()
    run_args['model']['trainable_parameters'] = predictor.trainable_parameters

    exp_logger = TensorBoardLogger(save_dir=cfg.run.dir, name=cfg.run.name)

    ########################################
    # training                             #
    ########################################

    early_stop_callback = EarlyStopping(
        monitor='val_winkler_at_90',
        patience=cfg.patience,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.run.dir,
        save_top_k=1,
        monitor='val_winkler_at_90',
        mode='min',
    )

    trainer = Trainer(max_epochs=cfg.epochs,
                      limit_train_batches=cfg.train_batches,
                      default_root_dir=cfg.run.dir,
                      logger=exp_logger,
                      accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                      devices=1,
                      gradient_clip_val=cfg.grad_clip_val,
                      callbacks=[early_stop_callback, checkpoint_callback])

    trainer.fit(predictor, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
    predictor.load_model(checkpoint_callback.best_model_path)

    ########################################
    # testing                              #
    ########################################

    predictor.freeze()
    # run validation one last time to save val error best model
    trainer.validate(predictor, dataloaders=dm.val_dataloader())
    trainer.test(predictor, dataloaders=dm.test_dataloader())

if __name__ == '__main__':
    exp = Experiment(run_fn=run_experiment, config_path='../config/corel',
                     config_name='default')
    res = exp.run()
    logger.info(res)