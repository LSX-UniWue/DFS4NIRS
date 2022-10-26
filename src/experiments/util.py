from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from joblib import dump
from scipy.special import expit, logit
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import RandomizedSearchCV

import experiment_args as args
from data.split_data import load_and_split_dataset


def to_probs(x):
    # Transform predictions for the idrc dataset back to original space
    return (expit(x) - 0.01) / 0.98


def to_logits(x):
    # Transform targets for the idrc dataset to the logit space
    return logit(0.98 * x + 0.01)


def random_search_with_cv(estimator: BaseEstimator,
                          params: Dict,
                          x_train: np.ndarray,
                          x_test: np.ndarray,
                          y_train: np.ndarray,
                          y_test: np.ndarray,
                          path_to_model: Path):
    cv = RandomizedSearchCV(estimator=estimator,
                            param_distributions=params,
                            n_iter=60,
                            scoring='neg_root_mean_squared_error',
                            n_jobs=5,
                            cv=5,
                            refit=True,
                            random_state=args.CV_RANDOM_STATE)
    cv.fit(x_train, y_train)
    results = pd.DataFrame.from_dict(cv.cv_results_)

    y_train_pred = cv.predict(x_train)
    y_test_pred = cv.predict(x_test)

    rmse_train = mse(y_train, y_train_pred, squared=False)
    rmse_test = mse(y_test, y_test_pred, squared=False)

    print(f'RMSE - train: {rmse_train:.4f}, test: {rmse_test:.4f}')

    path_to_model.mkdir(parents=True, exist_ok=True)
    dump(cv.best_estimator_, path_to_model / f'estimator.pkl')
    results.to_csv(path_to_model / 'cv_results.csv')

    return rmse_train, rmse_test


def get_train_test_data(dataset: str, path_to_datasets: Path):
    if dataset == args.IDRC_ARG:
        splits = load_and_split_dataset(path_to_datasets / args.PATH_TO_IDRC, external_test_set=True)
        x_train, x_test, y_train, y_test = splits

        # Transform targets from [0,1] to logit space for regression, rescaling is performed to prevent infinity values
        y_train = to_logits(y_train)
        y_test = to_logits(y_test)

        return x_train, x_test, y_train, y_test
    else:
        return load_and_split_dataset(path_to_datasets / args.DATASET_MAP[dataset])
