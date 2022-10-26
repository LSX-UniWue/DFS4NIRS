import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.pipeline import make_pipeline

import experiment_args as args
from experiments.util import get_train_test_data, to_probs
from models.preprocessors import Normalizer


def refit(path_to_results: Path,
          path_to_datasets: Path,
          dataset_name: str):
    x_train, x_test, y_train, y_test = get_train_test_data(dataset_name, path_to_datasets)

    scores = []
    for path_to_setting in path_to_results.glob('cfs_*'):
        for n_features in range(1, 11):
            path_to_runs = path_to_setting / 'Normalizer' / f'{n_features}_features' / '100_runs.csv'
            if not path_to_runs.exists():
                print(path_to_runs, ' does not exist! Stopping')
                continue

            runs = pd.read_csv(path_to_runs)
            selections = runs.iloc[:, -n_features:].values
            for idx, s in enumerate(selections):
                pipeline = make_pipeline(Normalizer(), LinearRegression())
                sidx = np.unique(s)
                pipeline.fit(x_train[:, sidx], y_train)
                y_test_pred = pipeline.predict(x_test[:, sidx])

                if dataset_name == args.IDRC_ARG:
                    rmse_test = mse(to_probs(y_test), to_probs(y_test_pred), squared=False)
                else:
                    rmse_test = mse(y_test, y_test_pred, squared=False)
                scores.append((path_to_setting.name, n_features, idx, runs.loc[idx]['RMSE (Test)'], 'DFS4NIRS E2E'))
                scores.append((path_to_setting.name, n_features, idx, rmse_test, 'DFS4NIRS (refit)'))
        print(f'Finished refits for {path_to_setting.name}')

    df = pd.DataFrame(data=scores, columns=['Setting', 'Features', 'Run', 'RMSE', 'Optimization'])
    df.to_csv(path_to_results / 'selection_refit_results.csv', index=False)


def main(argv=None):
    if argv is None:
        argv = sys.argv

    config = args.get_calibration_configuration(argv)

    for dataset in config.datasets:
        refit(path_to_results=config.path_to_results / args.DATASET_MAP[dataset].parent,
              path_to_datasets=config.path_to_datasets,
              dataset_name=dataset)


if __name__ == '__main__':
    sys.exit(main())
