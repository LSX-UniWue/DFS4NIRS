import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from skorch.callbacks import Callback


class TemperatureScheduler(Callback):
    """Adapts the temperature of the SelectionAndRegression module according to a pre-calculated schedule.

    Parameters
    ----------
    max_epochs: int
        number of training epochs
    temp_start: float
        temperature of the concrete distribution for the first training epoch
    temp_start: float
        temperature of the concrete distribution for the final training epoch
    variant: str
        type of the temperature schedule (one of linear or exponential)
    """

    def __init__(self, max_epochs: int, temp_start: float = 5, temp_end: float = 1e-2, variant: str = 'linear'):
        super().__init__()
        self.max_epochs = max_epochs
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.variant = variant
        self.schedule_ = self.create_schedule()

    def on_train_begin(self, net, **kwargs):
        self.schedule_ = self.create_schedule()

    def on_epoch_begin(self, net, **kwargs):
        epoch = net.history[-1]['epoch']
        temperature = self.schedule_[epoch - 1]
        net.module_.selector.temperature = temperature

    def create_schedule(self):
        t0, tn, e = self.temp_start, self.temp_end, self.max_epochs
        if self.variant == 'exponential':
            schedule = t0 * (tn / t0) ** (torch.arange(0, e) / e)
        else:
            schedule = torch.linspace(t0, tn, e)
        return schedule


class RegularizationThresholdScheduler(Callback):
    def __init__(self, max_epochs: int, threshold_start: float, threshold_end: float = 1):
        super().__init__()
        self.max_epochs = max_epochs
        self.threshold_start = threshold_start
        self.threshold_end = threshold_end
        self.schedule_ = self.create_schedule()

    def on_train_begin(self, net, **kwargs):
        self.schedule_ = self.create_schedule()

    def on_epoch_begin(self, net, **kwargs):
        epoch = net.history[-1]['epoch']
        threshold = self.schedule_[epoch - 1]
        net.criterion_.threshold = threshold

    def create_schedule(self):
        t0, tn, e = self.threshold_start, self.threshold_end, self.max_epochs
        schedule = t0 * (tn / t0) ** (torch.arange(0, e) / e)
        return schedule


class FeatureSelectionInitializer(Callback):
    def __init__(self):
        super().__init__()

    def on_train_begin(self, net, X=None, y=None, **kwargs):
        pcr = make_pipeline(PCA(n_components=net.module.regressor.in_features), LinearRegression())
        pcr.fit(X, y)
        components = pcr['pca'].components_
        weights, bias = pcr['linearregression'].coef_, pcr['linearregression'].intercept_

        net.module_.selector.log_alphas.data = torch.tensor(components)
        net.module_.regressor.weight.data = torch.tensor(weights)
        net.module_.regressor.bias.data = torch.tensor(bias)

        net.warm_start = True


class FinalRefit(Callback):
    def __init__(self):
        super().__init__()

    def on_train_end(self, net, X=None, y=None, **kwargs):
        x_selected = net.evaluation_step(X)[1]

        lr = LinearRegression()
        lr.fit(x_selected, y)
        weights, bias = lr.coef_, lr.intercept_

        net.module_.regressor.weight.data = torch.tensor(weights)
        net.module_.regressor.bias.data = torch.tensor(bias)


class FeatureSelectionLogger(Callback):
    def __init__(self):
        super().__init__()
        self.log_alphas_ = []
        self.temperatures_ = []

    def on_epoch_end(self, net, **kwargs):
        self.log_alphas_.append(net.module_.selector.log_alphas.clone())
        self.temperatures_.append(net.module_.selector.temperature.clone())
