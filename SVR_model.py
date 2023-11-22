import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     LeaveOneOut,
                                     KFold, ParameterGrid)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import skfda
from skfda.misc.hat_matrix import NadarayaWatsonHatMatrix
from skfda.ml.regression._kernel_regression import KernelRegression

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# from torchvision import datasets
# from torchvision import transforms

from tqdm.notebook import tqdm

import hydra
from omegaconf import DictConfig, ListConfig

import mlflow

import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)

def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f'{parent_name}.{k}', v)
            else:
                mlflow.log_param(f'{parent_name}.{k}', v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f'{parent_name}.{i}', v)

SVR_config_path = "/content/Machine_Learning_MIPT/config.yaml"

@hydra.main(config_path=SVR_config_path)
def SVR_model(cfg):
    model = SVR(degree=cfg.model.degree, C=cfg.model.C, epsilon=cfg.model.epsilon)
    
    mlflow.set_tracking_uri('file://' + hydra.utils.get_original_cwd() + '/mlruns')
    mlflow.set_experiment(cfg.mlflow.runname)

    with mlflow.start_run():
        log_params_from_omegaconf_dict(cfg)

        model.fit(X_train_array, y_train_array)
        pred = model.predict(X_test_array)
        mlflow.log_metrics(get_metrics(real=y_test_array, pred=pred))