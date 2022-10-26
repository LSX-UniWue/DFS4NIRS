import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.pipeline import Pipeline

import experiment_args as args
from experiments.util import get_train_test_data
from models.baseline_feature_selection import KBestRegression, FeatureSelector, RFERegression, SFSRegression

K_BEST_ARG = '--kbest'
RFE_ARG = '--rfe'
SFS_ARG = '--sfs'

FS_ARGS = {K_BEST_ARG, RFE_ARG, SFS_ARG}
FS_MAPPING = {K_BEST_ARG: KBestRegression, RFE_ARG: RFERegression, SFS_ARG: SFSRegression}


def calibrate_with_feature_selection(path_to_results: Path,
                                     estimator: BaseEstimator,
                                     fs_method: Callable[[int, BaseEstimator], FeatureSelector],
                                     spectra_train: np.ndarray,
                                     spectra_test: np.ndarray,
                                     targets_train: np.ndarray,
                                     targets_test: np.ndarray):
    path_to_results.mkdir(parents=True, exist_ok=True)

    scores = []
    for num_features in range(1, 11):
        feature_selection = fs_method(num_features, estimator)
        print(f'Starting calibration of {feature_selection.name} for n={num_features} features')

        path_to_model = path_to_results / feature_selection.name / f'{num_features}'
        path_to_model.mkdir(parents=True, exist_ok=True)
        feature_selection.calibrate(spectra_train, targets_train, path_to_model)

        y_train_pred = feature_selection.predict(spectra_train)
        y_test_pred = feature_selection.predict(spectra_test)

        rmse_train = mse(targets_train, y_train_pred, squared=False)
        rmse_test = mse(targets_test, y_test_pred, squared=False)

        print(f'RMSE - train: {rmse_train:.4f}, test: {rmse_test:.4f}')

        scores.append((num_features, rmse_train, rmse_test))

    scores = pd.DataFrame(data=scores, columns=['# Features', 'RMSE (train)', 'RMSE (test)'])
    scores.to_csv(path_to_results / feature_selection.name / 'scores.csv')


def main(argv=None):
    if argv is None:
        argv = sys.argv

    config = args.get_calibration_configuration(argv)
    fs_method = FS_ARGS.intersection(argv)
    if len(fs_method) != 1:
        print('Feature Selection method is ambiguous! Stopping calibration')
        return
    fs_method = FS_MAPPING[fs_method.pop()]

    for dataset in config.datasets:
        name = dataset.replace('--', '')
        preprocessor = config.preprocessor_init()
        estimator = Pipeline([(preprocessor.name, preprocessor), ('lr', LinearRegression())])

        x_train, x_test, y_train, y_test = get_train_test_data(dataset, config.path_to_datasets)

        path_to_results = config.path_to_results / name / 'baseline_feature_selection' / preprocessor.name
        calibrate_with_feature_selection(path_to_results=path_to_results,
                                         estimator=estimator,
                                         fs_method=fs_method,
                                         spectra_train=x_train,
                                         spectra_test=x_test,
                                         targets_train=y_train,
                                         targets_test=y_test)


if __name__ == '__main__':
    sys.exit(main())
