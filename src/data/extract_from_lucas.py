from pathlib import Path

import numpy as np
import pandas as pd
import shapefile

ID_COL = 'PointID'
ID_COL_METAS = 'Point_ID'
OC_COLUMN = 'OC_t'
CATEGORIES_COLUMN = 'LC0_Desc'
COLUMNS_NIR = [str(idx).replace('.0', '') for idx in np.arange(780, 2500, 0.5)]


def read_shapefile(path_to_sf: Path):
    sf = shapefile.Reader(str(path_to_sf))
    fields = [f[0] for f in sf.fields][1:]
    records = [r[:] for r in sf.records()]

    return pd.DataFrame(data=records, columns=fields)


def merge_spectra_into_single_csv(path_to_csv_files: Path,
                                  path_to_targets: Path,
                                  path_to_shapefile: Path):
    spectra = []
    for path in path_to_csv_files.glob('*.csv'):
        df = pd.read_csv(path)
        df = df[[ID_COL, *COLUMNS_NIR]].groupby(ID_COL).mean()
        spectra.append(df)

        print(f'{path.stem} loaded')
    spectra = pd.concat(spectra)

    targets = pd.read_excel(path_to_targets, engine='openpyxl')
    sf = read_shapefile(path_to_shapefile)
    metas = pd.merge(left=targets, right=sf, left_on=ID_COL_METAS, right_on=ID_COL_METAS,
                     how='inner', suffixes=['_t', '_s'])

    data = pd.merge(left=metas, right=spectra, left_on=ID_COL_METAS, right_on=ID_COL, how='inner')
    data = data[[*COLUMNS_NIR, OC_COLUMN, CATEGORIES_COLUMN]]
    data.columns = [*COLUMNS_NIR, 'target', 'category']

    data.to_csv(path_to_csv_files.parent / 'lucas.csv', index=False)


if __name__ == '__main__':
    path_to_spectra = Path().cwd() / 'data' / 'lucas' / 'spectra'
    path_to_lucas_targets = Path.cwd() / 'data' / 'lucas' / 'metas' / 'LUCAS_Topsoil_2015_20200323.xlsx'
    path_to_lucas_shapefile = Path.cwd() / 'data' / 'lucas' / 'metas' / 'shapefile' / 'LUCAS_Topsoil_2015_20200323'

    merge_spectra_into_single_csv(path_to_csv_files=path_to_spectra,
                                  path_to_targets=path_to_lucas_targets,
                                  path_to_shapefile=path_to_lucas_shapefile)
