import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import load
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error as mse

import experiment_args as args
from experiments.util import to_probs, get_train_test_data

torch.manual_seed(20211222)
torch.use_deterministic_algorithms(True)


def train_and_test_estimator(estimator: BaseEstimator,
                             spectra_train: np.ndarray,
                             spectra_test: np.ndarray,
                             targets_train: np.ndarray,
                             targets_test: np.ndarray,
                             transform_to_probs: bool = False):
    estimator[-1].warm_start = False
    estimator.fit(spectra_train, targets_train)

    y_train_pred = estimator.predict(spectra_train)
    y_test_pred = estimator.predict(spectra_test)
    if transform_to_probs:
        targets_train, targets_test = to_probs(targets_train), to_probs(targets_test)
        y_train_pred, y_test_pred = to_probs(y_train_pred), to_probs(y_test_pred)

    rmse_train = mse(targets_train, y_train_pred, squared=False)
    rmse_test = mse(targets_test, y_test_pred, squared=False)

    selected_features = estimator[-1].module_.selector.log_alphas.argmax(dim=1).detach().numpy()

    return rmse_train, rmse_test, selected_features


def recalibrate(path_to_results: Path,
                path_to_datasets: Path,
                dataset_name: str,
                num_refits: int = 100):
    x_train, x_test, y_train, y_test = get_train_test_data(dataset_name, path_to_datasets)
    x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)
    y_train, y_test = y_train[:, None].astype(np.float32), y_test[:, None].astype(np.float32)

    transform_to_probs = (dataset_name == args.IDRC_ARG)

    for p in path_to_results.glob('*_features'):
        num_features = p.name.split('_')[0]
        if not num_features.isdigit():
            print(f'Number of features can not be determined for {p.name}')
            continue
        num_features = int(num_features)

        if not (p / 'estimator.pkl').exists():
            print(f'No estimator found for {num_features} features')

        estimator = load(p / 'estimator.pkl')

        scores = []
        print(f'Starting {num_refits} refits for N={num_features} features')
        for i in range(num_refits):
            rmse_train, rmse_test, features = train_and_test_estimator(estimator,
                                                                       x_train, x_test, y_train, y_test,
                                                                       transform_to_probs)
            scores.append((i, rmse_train, rmse_test, *features))
        results = pd.DataFrame(data=scores, columns=['# Run', 'RMSE (Train)', 'RMSE (Test)', *np.arange(num_features)])
        print(results.describe())
        results.to_csv(p / f'{num_refits}_runs.csv')


def main(argv=None):
    if argv is None:
        argv = sys.argv

    config = args.get_calibration_configuration(argv)
    training_path = args.find_value_in_argv(argv, args.TRAINING_ARG)

    for dataset in config.datasets:
        name = dataset.replace('--', '')

        recalibrate(path_to_results=config.path_to_results / name / training_path,
                    path_to_datasets=config.path_to_datasets,
                    dataset_name=dataset)


if __name__ == '__main__':
    sys.exit(main())
