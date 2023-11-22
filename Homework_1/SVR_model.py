import numpy as np
from sklearn.svm import SVR

import hydra
from omegaconf import DictConfig, ListConfig

import mlflow
from pyngrok import ngrok

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


SVR_config_path = "./Homework_1"
X_train = np.load("./Homework_1/data/X_train_data.npy")
X_test = np.load("./Homework_1/data/X_test_data.npy")
y_train = np.load("./Homework_1/data/y_train_data.npy")
y_test = np.load("./Homework_1/data/y_test_data.npy")

@hydra.main(config_path=SVR_config_path, config_name="config")
def main(cfg):
    model = SVR(kernel=cfg.model.kernel, degree=cfg.model.degree, 
        gamma=cfg.model.gamma, C=cfg.model.C, epsilon=cfg.model.epsilon)

    with mlflow.start_run():
        log_params_from_omegaconf_dict(cfg)

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mlflow.log_metrics(get_metrics(real=y_test, pred=pred))

    get_ipython().system_raw("mlflow ui --port 5000 &")
    # Terminate open tunnels if exist.
    ngrok.kill()

    # My personal authtoken to open MLflow web page.
    NGROK_AUTH_TOKEN = "2YXCDq9gfZ1TqYNfes8aXEG0f8W_UWis1E3SUQ74wF9t1Pyk"
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)

    ngrok_tunnel = ngrok.connect(addr="5000", proto="http", bind_tls=True)
    print("MLflow Tracking UI:", ngrok_tunnel.public_url)


if __name__ == '__main__':
    main()