import pickle
from pathlib import Path

import numpy as np
import pandas as pd

RECIPES = ['R562', 'R568', 'R861', 'R862']


def generate_csv_from_pickle(path: Path):
    pkl_file = open(path, 'rb')
    data = pickle.load(pkl_file)

    wavenumbers = np.concatenate([data['wn1'], data['wn2']])

    df = []
    for r in RECIPES:
        absorbance1, absorbance2 = data[r]['X1'], data[r]['X2']
        turbidity = data[r]['Y']

        values = np.concatenate([absorbance1, absorbance2, turbidity], axis=1)
        _df = pd.DataFrame(data=values, columns=[*wavenumbers, 'target'])
        _df['category'] = [r] * len(_df)
        df.append(_df)

    df = pd.concat(df)
    df.to_csv(path.parent / (path.stem + '.csv'), index=False)


if __name__ == '__main__':
    path_to_data = Path.cwd() / 'data' / 'melamine' / 'Melamine_Dataset.pkl'
    generate_csv_from_pickle(path_to_data)
