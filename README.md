[![ICML](https://img.shields.io/badge/ICML-2025-blue.svg?style=flat-square)]()
[![arXiv](https://img.shields.io/badge/arXiv-2305.19183-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2502.09443)
[![PDF](https://img.shields.io/badge/%E2%87%A9-PDF-orange.svg?style=flat-square)](https://arxiv.org/pdf/2502.09443)


# Relational Conformal Prediction for Correlated Time Series (ICML 2025)

This repository contains the code to reproduce the experiments presented in the paper "*Relational Conformal Prediction for Correlated Time Series*".

**Authors**: [Andrea Cini](https://andreacini.github.io/), [Alexander Jenkins](a.jenkins21@imperial.ac.uk), Danilo Mandic, Cesare Alippi, [Filippo Maria Bianchi](https://sites.google.com/view/filippombianchi/home)

## Datasets

The datasets used in the experiments are provided by the <img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/tsl_logo.svg" width="25px" align="center"/> [**tsl**](https://torch-spatiotemporal.readthedocs.io/en/latest/) library. The CER-E dataset can be obtained for research purposes following the instructions at this [link](https://www.ucd.ie/issda/data/commissionforenergyregulationcer/).

## Configuration files

This project uses the <img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/logos/hydra.svg" width="25px" align="center"/> <a href="https://hydra.cc/">Hydra</a> library to handle the configuration of the hyperparameters and the other experimental settings.
The `config` directory stores the configuration files used by Hydra to run the experiments.

The folder `config/training` contains the configs for training the base models, while `config/corel` contains the configs to train the $\texttt{CoRel}$ and $\texttt{CoRNN}$ models.
To change a configuration, you can either modify the .yaml file directly or pass a different value to the args from the CLI (see example below).

## Requirements

To solve all dependencies, we recommend using [Anaconda or Miniconda](https://conda.io/projects/conda/en/latest/user-guide/install) to build an environment using the configuration specified on [conda_env.yml](conda_env.yml) by running the command:

```bash
conda env create -f conda_env.yml
```

Once the environment is created, activate it:

```bash
conda activate corel
```

## Experiments

The script used for the experiments in the paper is in the `experiments` folder.
The procedure is divided in two steps: 
1. train the base point predictor;
2. train the conformal predictor.

### Train the base model
The first step is to train base point predictor and compute the residuals for the calibration/validation set.
This is done by running [`run_base_model.py`](./experiments/run_base_model.py). 

For example run the following command to train an RNN base predictor on the METR-LA dataset. 
The `save_outputs` flag allows for saving the results that will be used for CP.

```bash
python -m experiments.run_base_model config=default model=rnn dataset=la save_outputs=true
```

The results will be saved inside the `/log` folder. For example

> logs/base/la/rnn/2025-06-08/12-15-13/

> [!CAUTION]
> The name of the folder will change each time you train the point predictor. Take note of the path to perform the next step.

### Train the conformal predictor
The next step is to train the conformal prediction model ($\texttt{CoRel}$ or $\texttt{CoRNN}$) by running [`run_corel.py`](./experiments/run_corel.py).
Note that here we need to pass the same directory created at the previous step, which contains the residuals of the calibration/validation set.
    
```bash
python -m experiments.run_corel config=default model=corel dataset=la src_dir="./logs/base/la/rnn/2025-06-08/12-15-13/"
```

## Reference

If you find this code useful please consider citing our paper:

```bibtex
@article{cini2025relational,
title        = {{Relational Conformal Prediction for Correlated Time Series}},
author       = {Cini, Andrea and Jenkins, Alexander and Mandic, Danilo and Alippi, Cesare and Bianchi, Filippo Maria},
journal      = {International Conference on Machine Learning},
year         = {2025}
}
```