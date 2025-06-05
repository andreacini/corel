import os
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from lib.datasets.gpvar import GPVARDataset
from lib.engines.base_predictor import BasePredictor
from lib.utils.data_utils import create_residuals_frame
from tsl import logger
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import MetrLA
from tsl.experiment import Experiment, NeptuneLogger
from tsl.metrics import torch_metrics

from lib import config
from tsl.nn.models import TransformerModel
from tsl.utils.casting import torch_to_numpy

from lib.nn.base import RNNModel, STGNNModel  # , MLPModel
from lib.datasets.air_quality import AirQuality


def get_model_class(model_str):
    # Basic models  #####################################################
    if model_str == 'rnn':
        model = RNNModel
    elif model_str == 'transformer':
        model = TransformerModel
    elif model_str == 'stgnn':
        model = STGNNModel
    else:
        raise NotImplementedError(f'Model "{model_str}" not available.')
    return model


def get_dataset(dataset_cfg):
    name = dataset_cfg.name
    if name == 'la':
        dataset = MetrLA()
    elif name == 'air':
        dataset = AirQuality()
    elif name == 'gpvar':
        dataset = GPVARDataset(**dataset_cfg.hparams, p_max=0)
    else:
        raise ValueError(f"Dataset {name} not available.")
    return dataset


def run_experiment(cfg: DictConfig):
    ########################################
    # data module                          #
    ########################################
    dataset = get_dataset(cfg.dataset)

    covariates = dict()
    if cfg.get('add_exogenous'):
        assert cfg.dataset.name not in {'gpvar'}
        # encode time of the day and use it as exogenous variable
        day_sin_cos = dataset.datetime_encoded('day').values
        weekdays = dataset.datetime_onehot('weekday').values
        covariates.update(u=np.concatenate([day_sin_cos, weekdays], axis=-1))

    if cfg.dataset.name in {'gpvar', 'toy', 'mso'}:
        ds_index = pd.Index(dataset.index)
        index_type = 'scalar'
    else:
        ds_index = dataset.index
        index_type = 'datetime'

    torch_dataset = SpatioTemporalDataset(index=ds_index,
                                          target=dataset.dataframe(),
                                          mask=dataset.mask,
                                          covariates=covariates,
                                          horizon=cfg.horizon,
                                          window=cfg.window,
                                          stride=cfg.stride,
                                          delay=cfg.get('delay', 0))

    if cfg.apply_scaler is False:
        transform = {}
    else:
        scale_axis = (0,) if cfg.get('scale_axis') == 'node' else (0, 1)
        transform = {
            'target': StandardScaler(axis=scale_axis)
        }

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=transform,
        splitter=dataset.get_splitter(**cfg.dataset.splitting),
        batch_size=cfg.batch_size,
        workers=cfg.workers
    )
    dm.setup()

    ########################################
    # training                             #
    ########################################

    # get adjacency matrix
    adj = dataset.get_connectivity(**cfg.dataset.connectivity,
                                   train_slice=dm.train_slice)
    dm.torch_dataset.set_connectivity(adj)

    ########################################
    # Create model                         #
    ########################################

    model_cls = get_model_class(cfg.model.name)

    d_exog = torch_dataset.input_map.u.shape[-1] if 'u' in torch_dataset else 0
    model_kwargs = dict(n_nodes=torch_dataset.n_nodes,
                        input_size=torch_dataset.n_channels,
                        exog_size=d_exog,
                        output_size=torch_dataset.n_channels,
                        weighted_graph=torch_dataset.edge_weight is not None,
                        window=torch_dataset.window,
                        horizon=torch_dataset.horizon)

    model_cls.filter_model_args_(model_kwargs)
    model_kwargs.update(cfg.model.hparams)

    ########################################
    # predictor                            #
    ########################################

    loss_fn = torch_metrics.MaskedMAE()

    log_metrics = {'mae': torch_metrics.MaskedMAE(),
                   'mse': torch_metrics.MaskedMSE(),
                   'mre': torch_metrics.MaskedMRE()}

    if cfg.dataset.name in ['la', 'bay']:
        multistep_metrics = {
            'mape': torch_metrics.MaskedMAPE(),
            'mae@15': torch_metrics.MaskedMAE(at=2),
            'mae@30': torch_metrics.MaskedMAE(at=5),
            'mae@60': torch_metrics.MaskedMAE(at=11),
        }
        log_metrics.update(multistep_metrics)

    # setup predictor
    predictor = BasePredictor(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=getattr(torch.optim, cfg.optimizer.name),
        optim_kwargs=dict(cfg.optimizer.hparams),
        loss_fn=loss_fn,
        metrics=log_metrics,
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
        monitor='val_mae',
        patience=cfg.patience,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.run.dir,
        save_top_k=1,
        monitor='val_mae',
        mode='min',
    )

    val_batches = .25

    trainer = Trainer(max_epochs=cfg.epochs,
                      limit_train_batches=cfg.train_batches,
                      limit_val_batches=val_batches,
                      default_root_dir=cfg.run.dir,
                      logger=exp_logger,
                      accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                      devices=1,
                      gradient_clip_val=cfg.grad_clip_val,
                      callbacks=[early_stop_callback, checkpoint_callback])

    load_model_path = cfg.get('load_model_path')
    if load_model_path is not None:
        predictor.load_model(load_model_path)
    else:
        trainer.fit(predictor, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())

    predictor.freeze()

    ########################################
    # compute residuals                    #
    ########################################

    output = trainer.predict(predictor, dataloaders=[dm.val_dataloader(),
                                                     dm.test_dataloader()])  # has size [[len_val], [len_test]]
    output = predictor.collate_prediction_outputs(output)  # has size [len_val + len_test]
    output = torch_to_numpy(output)
    y_hat, y_true, mask = (output['y_hat'], output['y'], output.get('mask', None))

    residuals = (y_true - y_hat).squeeze(-1)
    calib_indices = dm.valset.indices
    test_indices = dm.testset.indices

    # input covariates
    val_index = dm.torch_dataset.data_timestamps(calib_indices)['horizon']
    test_index = dm.torch_dataset.data_timestamps(test_indices)['horizon']

    # Remove residuals at the beginning and at the end of the time series that have less than window and horizon time steps respectively
    # The second dimension in the dataframe is nodes x horizon
    # Input: [samples, nodes, horizon]
    # Output: [filtered_samples, nodes x horizon]
    lagged_residuals = create_residuals_frame(residuals,
                                              np.concatenate([val_index, test_index], axis=0),
                                              channels_index=dataset._columns_multiindex(),
                                              horizon=cfg.horizon,
                                              idx_type=index_type)

    # concatenate indices and take the one corresponding to the last time step
    target_index = np.concatenate([val_index, test_index], axis=0)[:, 0]

    # combinations of nodex x horizon
    col_idx = [(c[0], f'{c[1]}_{i}') for c in dataset._columns_multiindex() for i in range(cfg.horizon)]

    # create a dataframe with the residuals arranged in shape [samples, nodes x horizon]
    target_df = pd.DataFrame(data=rearrange(residuals, "t h n ... -> t (n ... h)"),
                             index=target_index,
                             columns=pd.MultiIndex.from_tuples(col_idx))
    if mask is not None:
        mask_df = pd.DataFrame(data=rearrange(mask, "t h n ... -> t (n ... h)"),
                               index=target_index,
                               columns=pd.MultiIndex.from_tuples(col_idx))
    else:
        mask_df = None

    # filter calib and test indices
    valid_input_indices = torch_dataset.index.get_indexer(lagged_residuals.index)
    valid_target_indices = torch_dataset.index.get_indexer(target_index)

    ########################################
    # save residuals for CP                #
    ########################################
    if cfg.save_outputs:
        lagged_residuals.to_hdf(os.path.join(cfg.run.dir,
                                             'residuals.h5'),
                                key='input')
        target_df.to_hdf(os.path.join(cfg.run.dir,
                                             'residuals.h5'),
                                key='target')
        if mask_df is not None:
            mask_df.to_hdf(os.path.join(cfg.run.dir,
                                             'residuals.h5'),
                                key='target_mask')
        np.savez(os.path.join(cfg.run.dir,'indices.npz'),
                 calib_indices=calib_indices,
                 test_indices=test_indices,
                 valid_input_indices=valid_input_indices,
                 valid_target_indices=valid_target_indices)

    return 'done'


if __name__ == '__main__':
    exp = Experiment(run_fn=run_experiment, config_path='../config/training',
                     config_name='default')
    res = exp.run()
    logger.info(res)