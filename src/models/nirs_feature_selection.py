from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import numpy as np
from joblib import dump
from oct2py import octave
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression


class ExternalFeatureSelection(ABC):
    name: str
    selected_features: np.ndarray
    models: Dict[int, BaseEstimator]

    def __init__(self, max_num_features: int, path_to_lib: Path):
        octave.addpath(str(path_to_lib))
        self.max_num_features = max_num_features

    @abstractmethod
    def perform_selection(self, x_train: np.ndarray, y_train: np.ndarray):
        ...

    def calibrate(self, x_train: np.ndarray, y_train: np.ndarray, path_to_calibration_result: Path = None):
        self.perform_selection(x_train, y_train)

        models = {}
        for i in range(1, self.max_num_features + 1):
            model = LinearRegression()
            model.fit(x_train[:, self.selected_features[:i]], y_train)
            models[i] = model
        self.models = models

        if path_to_calibration_result is not None:
            path_to_calibration_result.mkdir(parents=True, exist_ok=True)
            dump(self, path_to_calibration_result / 'selector.pkl')

    def predict(self, x: np.ndarray, num_features: int):
        y_pred = self.models[num_features].predict(x[:, self.selected_features[:num_features]])
        return y_pred


class VIP(ExternalFeatureSelection):
    name = 'vip'

    def perform_selection(self, x_train: np.ndarray, y_train: np.ndarray):
        results_pls = octave.feval('pls.m', x_train, y_train[:, None], 64)

        self.selected_features = results_pls.VIP[0].argsort()[-10:][::-1]


class MCUVE(ExternalFeatureSelection):
    name = 'mcuve'

    def perform_selection(self, x_train: np.ndarray, y_train: np.ndarray):
        result = octave.feval('mcuvepls.m', x_train, y_train[:, None], 64, 'center', 100)
        self.selected_features = result.SortedVariable[0].argsort()[:10]


class RF(ExternalFeatureSelection):
    name = 'rf'

    def perform_selection(self, x_train: np.ndarray, y_train: np.ndarray):
        result = octave.feval('randomfrog_pls.m', x_train, y_train[:, None], 64)
        self.selected_features = result.Vtop10[0].astype(int)
