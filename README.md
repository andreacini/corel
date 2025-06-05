# Relational Conformal Prediction for Correlated Time Series (ICML 2025)

This repository contains the code for the reproducibility of the experiments presented in the paper "Relational Conformal Prediction for Correlated Time Series".

**Authors**: [Andrea Cini](mailto:andrea.cini@usi.ch), Alexander Jenkins, Danilo Mandic, Cesare Alippi, Filippo Maria Bianchi

## Datasets

The datasets used in the experiments are provided by the `tsl` library. The CER-E dataset can be obtained for research purposes following the instructions at this [link](https://www.ucd.ie/issda/data/commissionforenergyregulationcer/).

## Configuration files

The `config` directory stores the configuration files used to run the experiments.

## Requirements

To solve all dependencies, we recommend using Anaconda and the provided environment configuration by running the command:

```bash
conda env create -f conda_env.yml
conda activate corel
```

## Experiments

The script used for the experiments in the paper is in the `experiments` folder.

* `run_base_model.py` is used to train the base point predictor. For example run the following command to train an RNN on the METR-LA dataset. The `save_outputs` flag allows for saving the results that will be used for CP.

	```
	python -m experiments.run_base_model config=default model=rnn dataset=la save_outputs=true
	```

* `run_corel.py` is used to train the conformal prediction model. It requires a directory containing the saved residuals for calibration.
    
	```
	python -m experiments.run_corel config=default model=corel dataset=la src_dir=???
	```
 

## Bibtex reference

If you find this code useful please consider citing our paper:

```
@article{cini2025relational,
title        = {{Relational Conformal Prediction for Correlated Time Series}},
author       = {Cini, Andrea and Jenkins, Alexander and Mandic, Danilo and Alippi, Cesare and Bianchi, Filippo Maria},
journal      = {International Conference on Machine Learning},
year         = {2025}
}
```