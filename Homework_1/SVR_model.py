import numpy as np
from sklearn.svm import SVR

import hydra
from omegaconf import DictConfig, ListConfig

import mlflow

from data_preparation import get_metrics

import random
random.seed(42)
np.random.seed(42)


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


SVR_config_path = "/content/Machine_Learning_MIPT/Homework_1"
X_train = np.load("/content/Machine_Learning_MIPT/Homework_1/data/X_train.npy", 
    allow_pickle=True)
X_test = np.load("/content/Machine_Learning_MIPT/Homework_1/data/X_test.npy", 
    allow_pickle=True)
y_train = np.load("/content/Machine_Learning_MIPT/Homework_1/data/y_train.npy", 
    allow_pickle=True)
y_test = np.load("/content/Machine_Learning_MIPT/Homework_1/data/y_test.npy", 
    allow_pickle=True)

mlflow.set_experiment("Support Vector Regression")

@hydra.main(config_path=SVR_config_path, config_name="config")
def main(cfg):
    with mlflow.start_run():
        model = SVR(kernel=cfg.model.kernel, degree=cfg.model.degree,
                    gamma=cfg.model.gamma, C=cfg.model.C,
                    epsilon=cfg.model.epsilon)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        (mse, mae, r2) = get_metrics(real=y_test, pred=pred)

        mlflow.log_param("kernel", cfg.model.kernel)
        mlflow.log_param("degree", cfg.model.degree)
        mlflow.log_param("gamma", cfg.model.gamma)
        mlflow.log_param("C", cfg.model.C)
        mlflow.log_param("epsilon", cfg.model.epsilon)

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mar", mae)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(model, "SVR")


if __name__ == '__main__':
    main()