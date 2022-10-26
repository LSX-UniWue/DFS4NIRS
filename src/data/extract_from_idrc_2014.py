from pathlib import Path

import numpy as np
import pandas as pd


def get_temperature(df):
    temperature_celcius = df['Temperature degree c'].values
    return temperature_celcius + 273.15


def get_pressure(df):
    pressure_psi = df['Pressure psi'].values
    return pressure_psi * 6895 * 1e-6


def get_path_length(df):
    temperature_kelvin = get_temperature(df)
    pressure_mega_pascal = get_pressure(df)
    return 0.8801 + 0.000402065 * temperature_kelvin + 0.00060493 * pressure_mega_pascal


def transmittance_to_absorbance(transmittance):
    return -np.log10(transmittance)


def absorbance_to_transmittance(absorbance):
    return 10 ** (-absorbance)


def generate_csv_from_raw_data(path: Path, filename: str):
    """Load raw spectra and reference values and performs path length correction according to
    Bogomolov et al., Summary of the 2014 IDRC software shoot-out, NIR News 26, 2015
    """

    df = pd.read_csv(path, index_col=0)

    absorbance = transmittance_to_absorbance(df.values[:, 3:])
    path_length = get_path_length(df)
    transmittance = absorbance_to_transmittance(absorbance / path_length[:, None])

    targets = df['Reference value'].values

    data = np.concatenate([transmittance, targets[:, None]], axis=1)
    df = pd.DataFrame(data=data, columns=[*df.columns[3:], 'target'])
    df.to_csv(path.parent / f'{filename}.csv', index=False)


if __name__ == '__main__':
    path_to_data = Path.cwd() / 'data' / 'idrc'

    generate_csv_from_raw_data(path_to_data / 'DataSet1_Cal.csv', filename='idrc_train')
    generate_csv_from_raw_data(path_to_data / 'DataSet1_Test.csv', filename='idrc_test')
