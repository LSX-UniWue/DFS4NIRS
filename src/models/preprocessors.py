import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
from sklearn.base import BaseEstimator, TransformerMixin


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, name: str = None, split_at: int = -1, params=None):
        self.name = name
        self.split_at = split_at
        self.params = params

    def fit(self, spectra, targets=None):
        return self

    def transform(self, spectra):
        if self.split_at > 0:
            s1, s2 = spectra[:, :self.split_at], spectra[:, self.split_at:]
            t1, t2 = self(s1), self(s2)
            transformed = np.concatenate([t1, t2], axis=1)
        else:
            transformed = self(spectra)
        return transformed

    def __call__(self, spectra):
        return spectra


class Identity(Preprocessor):
    """Identity mapping

    Allows the Either preprocessor to select no preprocessing method.
    """

    def __init__(self, split_at: int = -1):
        super().__init__(name='identity', split_at=-1)  # identity does not change for a splitted spectra

    def __call__(self, spectra):
        return spectra


class Normalizer(Preprocessor):
    """Standard Scaler

    Standardizes each feature to zero mean and unit variance given the sample statistics from the training dataset.
    """

    def __init__(self, split_at: int = -1):
        super().__init__(name='Normalizer', split_at=-1)  # normalization does not change for split spectra
        self.mean = 0
        self.std = 1

    def fit(self, spectra, targets=None):
        self.mean = np.mean(spectra, axis=0, keepdims=True)
        self.std = np.std(spectra, axis=0, keepdims=True)
        return self

    def __call__(self, spectra):
        return (spectra - self.mean) / (1e-8 + self.std)


class SNV(Preprocessor):
    """Standard Normal Variate (SNV)

    Standardizes each spectrum individually to zero mean and unit variance.
    """

    def __init__(self, split_at: int = -1):
        super().__init__(name='snv', split_at=split_at)

    def __call__(self, spectra):
        means, stds = spectra.mean(axis=1, keepdims=True), spectra.std(axis=1, keepdims=True)
        return (spectra - means) / (1e-8 + stds)


class CR(Preprocessor):
    """Continuum Removal (CR)

    Removes the convex hull of a spectrum; also known as rubber band baseline correction
    """

    def __init__(self, split_at: int = -1):
        super().__init__(name='cr', split_at=split_at)

    def __call__(self, spectra):
        return np.array([self._cr(x) for x in spectra])

    @staticmethod
    def _cr(x):
        """Adapted from Quasars spectroscopy package for the Orange Data Mining software:
        https://github.com/Quasars/orange-spectroscopy/blob/cb10be9daee06fe8703af63dc28c3a9a2a88e24c/orangecontrib/spectroscopy/preprocess/__init__.py#L175

        TODO: check licence (GPL3+)
        """
        position = np.arange(len(x))
        data = np.column_stack((position, x))

        try:
            vertices = ConvexHull(data).vertices
        except (QhullError, ValueError):
            baseline = np.zeros_like(x)
        else:
            vertices = np.roll(vertices, -vertices.argmin())
            vertices = vertices[:vertices.argmax() + 1]
            baseline = interp1d(data[vertices, 0], data[vertices, 1], bounds_error=False)(position)

        return x - baseline


class SGF(Preprocessor):
    """
    Savitzky-Golay Filter (SGF)

    Performs local smoothing over the input data.
    """

    def __init__(self,
                 window_length: int = 11,
                 poly_order: int = 2,
                 derivative: int = 0,
                 split_at: int = -1):
        super().__init__(name='SGF', split_at=split_at)

        if derivative > poly_order:
            derivative = poly_order
        self.window_length = window_length
        self.poly_order = poly_order
        self.derivative = derivative

    def __call__(self, spectra):
        return savgol_filter(x=spectra,
                             window_length=self.window_length,
                             polyorder=self.poly_order,
                             deriv=self.derivative,
                             axis=-1)


class NIRPreprocessor(Preprocessor):
    def __init__(self, method: str = None, split_at: int = -1):
        super().__init__(name='NIRPreprocessor', split_at=split_at)
        preprocessors = [Identity(split_at=split_at),
                         Normalizer(split_at=split_at),
                         SNV(split_at=split_at),
                         CR(split_at=split_at)]

        self.preprocessors = {pp.name: pp for pp in preprocessors}
        names = list(self.preprocessors.keys())
        self.method = method if method in names else names[0]
        self.params = {'method': names}

    def fit(self, spectra, target=None):
        self.preprocessors = {name: pp.fit(spectra) for name, pp in self.preprocessors.items()}
        return self

    def __call__(self, spectra):
        return self.preprocessors[self.method](spectra)
