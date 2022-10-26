import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error as mse

import experiment_args as args
from experiments.util import get_train_test_data
from models.nirs_feature_selection import VIP, MCUVE, RF

VIP_ARG = '--vip'
MCUVE_ARG = '--mcuve'
RF_ARG = '--rf'

MODEL_ARGS = {VIP_ARG, MCUVE_ARG, RF_ARG}
MODEL_MAPPING = {VIP_ARG: VIP, MCUVE_ARG: MCUVE, RF_ARG: RF}


def main(argv=None):
    if argv is None:
        argv = sys.argv

    config = args.get_calibration_configuration(argv)

    method = MODEL_ARGS.intersection(argv)
    if len(method) != 1:
        print('Selected method is ambiguous. Stopping calibration!')
        return
    method = MODEL_MAPPING[method.pop()]

    path_to_lib = Path(args.find_value_in_argv(argv, args.EXTERNAL_LIB_ARG))
    if (path_to_lib is None) or (not path_to_lib.exists()):
        print('No existing path to external lib given. Calibration will not be performed!')
        return

    for dataset in config.datasets:
        preprocessor = config.preprocessor_init()
        model = method(path_to_lib=path_to_lib, max_num_features=10)

        x_train, x_test, y_train, y_test = get_train_test_data(dataset, config.path_to_datasets)

        preprocessor.fit(x_train)
        x_train, x_test = preprocessor.transform(x_train), preprocessor.transform(x_test)

        path_to_results = config.path_to_results / dataset.replace('-', '') / 'nirs_fs' / model.name
        model.calibrate(x_train, y_train, path_to_calibration_result=path_to_results)

        scores = []
        for num_features in range(1, 11):
            y_train_pred = model.predict(x_train, num_features=num_features)
            y_test_pred = model.predict(x_test, num_features=num_features)

            rmse_train = mse(y_train, y_train_pred, squared=False)
            rmse_test = mse(y_test, y_test_pred, squared=False)

            print(f'{num_features} RMSE - train: {rmse_train:.4f}, test: {rmse_test:.4f}')

            scores.append((num_features, rmse_train, rmse_test))

        scores = pd.DataFrame(data=scores, columns=['# Features', 'RMSE (train)', 'RMSE (test)'])
        scores.to_csv(path_to_results / 'scores.csv')


if __name__ == '__main__':
    sys.exit(main())
