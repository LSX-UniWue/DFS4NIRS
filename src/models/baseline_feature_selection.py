from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, parallel_backend
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectKBest, RFE, SequentialFeatureSelector, f_regression, mutual_info_regression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline

SFS_RANDOM_STATE = 20211109


class FeatureSelector(ABC):
    name: str

    @abstractmethod
    def calibrate(self, x_train: np.ndarray, y_train: np.ndarray, path_to_calibration_result: Path):
        ...

    @abstractmethod
    def predict(self, x: np.ndarray):
        ...


class KBestRegression(FeatureSelector):

    def __init__(self, num_features: int, estimator: BaseEstimator):
        self.name = 'kbest'

        if isinstance(estimator, Pipeline):
            model = Pipeline([(self.name, SelectKBest(k=num_features)), *estimator.steps])
        else:
            model = Pipeline([(self.name, SelectKBest(k=num_features)), ('estimator', estimator)])
        self.model = model
        self.params = {'kbest__score_func': [f_regression, mutual_info_regression]}

    def calibrate(self,
                  x_train: np.ndarray,
                  y_train: np.ndarray,
                  path_to_calibration_result: Path):
        cv = GridSearchCV(estimator=self.model,
                          param_grid=self.params,
                          scoring='neg_root_mean_squared_error',
                          n_jobs=5,
                          cv=5,
                          refit=True)
        cv.fit(x_train, y_train)
        results = pd.DataFrame.from_dict(cv.cv_results_)

        self.model = cv.best_estimator_
        dump(self.model, path_to_calibration_result / 'estimator.pkl')
        results.to_csv(path_to_calibration_result / 'cv_results.csv')

    def predict(self, x: np.ndarray):
        return self.model.predict(x)


class RFERegression(FeatureSelector):
    def __init__(self, num_features: int, estimator: BaseEstimator):
        self.name = 'rfe'

        importance = f'named_steps.{estimator.steps[-1][0]}.coef_' if isinstance(estimator, Pipeline) else 'coef_'
        self.model = RFE(estimator=estimator, n_features_to_select=num_features, importance_getter=importance)
        self.params = {'step': range(1, 11)}

    def calibrate(self, x_train: np.ndarray, y_train: np.ndarray, path_to_calibration_result: Path):
        with parallel_backend(backend='loky', n_jobs=4):
            self.model.fit(x_train, y_train)
        dump(self.model, path_to_calibration_result / 'estimator.pkl')

    def predict(self, x: np.ndarray):
        return self.model.predict(x)


class SFSRegression(FeatureSelector):
    def __init__(self, num_features: int, estimator: BaseEstimator):
        self.name = 'sfs'

        self.selector = SequentialFeatureSelector(estimator=estimator,
                                                  n_features_to_select=num_features,
                                                  direction='forward',
                                                  scoring='neg_root_mean_squared_error',
                                                  cv=KFold(n_splits=5, shuffle=True, random_state=SFS_RANDOM_STATE),
                                                  n_jobs=5)
        self.model = estimator

    def calibrate(self, x_train: np.ndarray, y_train: np.ndarray, path_to_calibration_result: Path):
        x_train_selection = self.selector.fit_transform(x_train, y_train)
        self.model.fit(x_train_selection, y_train)
        dump(self.selector, path_to_calibration_result / 'estimator.pkl')

    def predict(self, x: np.ndarray):
        x_selected = self.selector.transform(x)
        return self.model.predict(x_selected)
