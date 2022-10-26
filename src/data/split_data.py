from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE_TRAIN_TEST_SPLIT = 20211104


def load_dataset_from_csv_file(path_to_csv_file: Path):
    """Reads data from a csv file that contains the dataset samples and creates corresponding numpy arrays

    Parameters
    ----------
    path_to_csv_file: pathlib.Path
        location of the csv file containing the dataset

    Returns
    -------
    s, t, c: Tuple of numpy.ndarray
        Spectra, target values and categories
    """
    df = pd.read_csv(path_to_csv_file, header=0)

    spectra = df.iloc[:, :-2].values
    targets = df['target'].values
    categories = df['category'] if 'category' in df.columns else None

    return spectra, targets, categories


def load_and_split_dataset(path_to_csv_file: Path,
                           test_size: float = 0.25,
                           external_test_set: bool = False,
                           random_state: int = RANDOM_STATE_TRAIN_TEST_SPLIT):
    """Loads a dataset from a csv file into numpy array with the random/given train-test splits.

    Parameters
    ----------
    path_to_csv_file: pathlib.Path
        Path to the csv file containing the dataset
    test_size: float
        Relative size of the test set from the overall dataset (is not used when external_test_set==TRUE)
    external_test_set: bool:
        External test set with filename suffix '_test' is loaded (True) instead of a random test set (False)
    random_state: int
        Random state for the sklearn train_test_split function

    Returns
    --------
    x_train, x_test, y_train, y_test: Tuple of numpy.ndarray
        Split spectra and targets
    """

    if external_test_set:
        df_train = pd.read_csv(path_to_csv_file)
        df_test = pd.read_csv(path_to_csv_file.parent / path_to_csv_file.name.replace('train', 'test'))

        x_train, y_train = df_train.iloc[:, :-1].values, df_train['target'].values
        x_test, y_test = df_test.iloc[:, :-1].values, df_test['target'].values
    else:
        spectra, targets, categories = load_dataset_from_csv_file(path_to_csv_file)
        splits = train_test_split(spectra, targets, test_size=test_size, stratify=categories, random_state=random_state)
        x_train, x_test, y_train, y_test = splits

    return x_train, x_test, y_train, y_test
