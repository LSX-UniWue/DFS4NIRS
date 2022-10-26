from pathlib import Path

import pandas as pd

TARGET_COLUMN = 'TOTAL'


def merge_csv_into_single_file(path: Path):
    df_properties = pd.read_csv(path / 'diesel_prop.csv', skiprows=8, index_col=1).dropna(axis=1, how='all')
    df_spectra = pd.read_csv(path / 'diesel_spec.csv', skiprows=9, index_col=1).dropna(axis=1, how='all')
    df_targets = df_properties[TARGET_COLUMN].dropna()
    df_targets.name = 'target'

    df = pd.merge(left=df_spectra, right=df_targets, left_index=True, right_index=True, how='inner')
    df.to_csv(path / 'swri_diesel.csv', index=False)


if __name__ == '__main__':
    path_to_data = Path.cwd() / 'data' / 'swri_diesel'
    merge_csv_into_single_file(path_to_data)
