from dataclasses import dataclass
from pathlib import Path
from typing import List, Callable

from models.preprocessors import Preprocessor, SNV, Normalizer

SYNTHETIC_ARG = '--synthetic'
SYNTHETIC_SPECTRA_ARG = '--synthetic_spectra'
MELAMINE_ARG = '--melamine'
LUCAS_ARG = '--lucas'
IDRC_ARG = '--idrc'
SWRI_ARG = '--swri'

SNV_ARG = '--snv'
NORMALIZE_ARG = '--normalize'

CFS_INITIALIZE_ARG = '--initialize_with_pcr'
CFS_REGULARIZE_ARG = '--perform_regularization'
CFS_REFIT_ARG = '--refit_after_training'
CFS_EXP_TEMP_SCHEDULE_ARG = '--exp_temperature_schedule'
CFS_EPOCHS_ARG = '--cfs_epochs='

PATH_ARG = '--datasets_path='
RESULT_ARG = '--results_path='
TRAINING_ARG = '--training_path='
EXTERNAL_LIB_ARG = '--external_lib_path='

PATH_TO_SYNTHETIC = Path('synthetic') / 'synthetic.csv'
PATH_TO_SYNTHETIC_SPECTRA = Path('synthetic_spectra') / 'spectra.csv'
PATH_TO_MELAMINE = Path('melamine') / 'Melamine_Dataset.csv'
PATH_TO_LUCAS = Path('lucas') / 'lucas.csv'
PATH_TO_IDRC = Path('idrc') / 'idrc_train.csv'
PATH_TO_SWRI = Path('swri_diesel') / 'swri_diesel.csv'

DATASET_ARGS = {SYNTHETIC_ARG, SYNTHETIC_SPECTRA_ARG, MELAMINE_ARG, LUCAS_ARG, IDRC_ARG, SWRI_ARG}
DATASET_MAP = {
    SYNTHETIC_ARG: PATH_TO_SYNTHETIC,
    SYNTHETIC_SPECTRA_ARG: PATH_TO_SYNTHETIC_SPECTRA,
    MELAMINE_ARG: PATH_TO_MELAMINE,
    LUCAS_ARG: PATH_TO_LUCAS,
    IDRC_ARG: PATH_TO_IDRC,
    SWRI_ARG: PATH_TO_SWRI
}

PP_ARGS = {SNV_ARG, NORMALIZE_ARG}

CV_RANDOM_STATE = 20211202


@dataclass
class CalibrationConfiguration:
    datasets: List[str]
    path_to_datasets: Path
    path_to_results: Path
    preprocessor_init: Callable[..., Preprocessor]


def get_calibration_configuration(argv):
    datasets_for_calibration = list(DATASET_ARGS.intersection(argv))
    if len(datasets_for_calibration) < 1:
        print('No known dataset given. Calibration will not be performed!')
        return

    path_to_datasets = Path(find_value_in_argv(argv, PATH_ARG))
    path_to_results = Path(find_value_in_argv(argv, RESULT_ARG, default=''))

    if (path_to_datasets is None) or (not path_to_datasets.exists()):
        print('No existing path to datasets given. Calibration will not be performed!')
        return

    pp_args = PP_ARGS.intersection(argv)
    preprocessor_init = SNV
    if len(pp_args) != 1:
        print('Preprocessing method ambiguous. Using SNV')
    else:
        preprocessor_init = Normalizer if (pp_args.pop() == NORMALIZE_ARG) else SNV

    return CalibrationConfiguration(datasets=datasets_for_calibration,
                                    path_to_datasets=path_to_datasets,
                                    path_to_results=path_to_results,
                                    preprocessor_init=preprocessor_init)


def find_value_in_argv(argv, arg, default=None):
    for v in argv:
        if arg in v:
            return v.replace(arg, '')
    return default
