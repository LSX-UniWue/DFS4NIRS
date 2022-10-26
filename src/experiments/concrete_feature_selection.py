import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import loguniform, uniform
from sklearn.pipeline import make_pipeline
from skorch import NeuralNetRegressor
from torch.optim import Adam

import experiment_args as args
from experiments.util import random_search_with_cv, get_train_test_data
from models.concrete_regressor import SelectionAndRegression
from models.preprocessors import Preprocessor, Normalizer
from utils.callbacks import TemperatureScheduler, FeatureSelectionLogger, FeatureSelectionInitializer, FinalRefit, \
    RegularizationThresholdScheduler
from utils.loss import RegularizationLoss

torch.manual_seed(20211213)
torch.use_deterministic_algorithms(True)


def calibrate_concrete_model(path_to_results: Path,
                             spectra_train: np.ndarray,
                             spectra_test: np.ndarray,
                             targets_train: np.ndarray,
                             targets_test: np.ndarray,
                             epochs: int = 100,
                             preprocessor: Preprocessor = Normalizer(),
                             exponential_schedule: bool = False,
                             initialize_with_pcr: bool = False,
                             perform_regularization: bool = False,
                             refit_after_training: bool = False):
    path_to_results.mkdir(parents=True, exist_ok=True)

    scores = []
    for num_features in range(1, 11):
        print(f'Starting calibration for n={num_features} features')
        module = SelectionAndRegression(input_dim=spectra_train.shape[1], feature_dim=num_features)

        callbacks = [('temperature', TemperatureScheduler(max_epochs=epochs,
                                                          temp_start=5,
                                                          temp_end=1e-2,
                                                          variant='exponential' if exponential_schedule else 'linear')),
                     ('logger', FeatureSelectionLogger())]
        if initialize_with_pcr:
            callbacks.append(('initializer', FeatureSelectionInitializer()))
        if refit_after_training:
            callbacks.append(('refit', FinalRefit()))
        if perform_regularization:
            scheduler = RegularizationThresholdScheduler(max_epochs=epochs,
                                                         threshold_start=num_features)
            callbacks.append(('reg_scheduler', scheduler))

        net = NeuralNetRegressor(module=module,
                                 criterion=RegularizationLoss,
                                 criterion__strength=0,  # (strength=0) => mse
                                 criterion__threshold=num_features,
                                 optimizer=Adam,
                                 optimizer__weight_decay=0,
                                 batch_size=64,
                                 max_epochs=epochs,
                                 callbacks=callbacks,
                                 train_split=None,
                                 verbose=1)

        params = {'neuralnetregressor__lr': loguniform(1e-2, 1e0),
                  'neuralnetregressor__callbacks__temperature__temp_start': uniform(5, 5),
                  'neuralnetregressor__optimizer__weight_decay': loguniform(1e-4, 1e-2)}
        if perform_regularization:
            params['neuralnetregressor__criterion__strength'] = loguniform(1e-2, 1e0)
            params['neuralnetregressor__callbacks__reg_scheduler__threshold_start'] =\
                uniform(num_features/2, num_features/2)

        model = make_pipeline(preprocessor, net)
        rmse_train, rmse_test = random_search_with_cv(estimator=model,
                                                      params=params,
                                                      x_train=spectra_train,
                                                      x_test=spectra_test,
                                                      y_train=targets_train,
                                                      y_test=targets_test,
                                                      path_to_model=path_to_results / f'{num_features}_features')
        scores.append((num_features, rmse_train, rmse_test))

    scores = pd.DataFrame(data=scores, columns=['# Features', 'RMSE (train)', 'RMSE (test)'])
    scores.to_csv(path_to_results / 'scores.csv')


def main(argv=None):
    if argv is None:
        argv = sys.argv

    config = args.get_calibration_configuration(argv)

    init = (args.CFS_INITIALIZE_ARG in sys.argv)
    reg = (args.CFS_REGULARIZE_ARG in sys.argv)
    refit = (args.CFS_REFIT_ARG in sys.argv)
    exponential_schedule = (args.CFS_EXP_TEMP_SCHEDULE_ARG in sys.argv)
    epochs = args.find_value_in_argv(argv, args.CFS_EPOCHS_ARG, default='100')

    if epochs.isdigit():
        epochs = int(epochs)
    else:
        print('Can not infer epochs. Taking default value: 100')
        epochs = 100

    dir_name = ''.join(['cfs',
                        '_init' if init else '',
                        '_exp' if exponential_schedule else '',
                        '_reg' if reg else '',
                        '_refit' if refit else '',
                        f'_{epochs}'])

    for dataset in config.datasets:
        name = dataset.replace('--', '')
        preprocessor = config.preprocessor_init(split_at=225 if dataset == args.MELAMINE_ARG else -1)

        x_train, x_test, y_train, y_test = get_train_test_data(dataset, config.path_to_datasets)
        x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)
        y_train, y_test = y_train[:, None].astype(np.float32), y_test[:, None].astype(np.float32)

        print(f'Starting calibration for {name}')
        calibrate_concrete_model(path_to_results=config.path_to_results / name / dir_name / preprocessor.name,
                                 spectra_train=x_train,
                                 spectra_test=x_test,
                                 targets_train=y_train,
                                 targets_test=y_test,
                                 epochs=epochs,
                                 preprocessor=preprocessor,
                                 exponential_schedule=exponential_schedule,
                                 initialize_with_pcr=init,
                                 perform_regularization=reg,
                                 refit_after_training=refit)


if __name__ == '__main__':
    sys.exit(main())
