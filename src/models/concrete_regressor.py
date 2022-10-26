import torch
import torch.nn.functional as F
from torch.nn import Parameter, Module, Linear


class ConcreteFeatureSelection(Module):
    def __init__(self, input_dim: int, output_dim: int, temperature: float = 1e1):
        super().__init__()
        self.log_alphas = Parameter(torch.ones(output_dim, input_dim))
        self.temperature = temperature

    def forward(self, inputs):
        probabilities = F.gumbel_softmax(logits=self.log_alphas, tau=self.temperature, hard=not self.training)
        if self.training:
            features = inputs @ probabilities.T
        else:
            features = inputs[:, self.log_alphas.argmax(dim=1)]
        return features, probabilities


class SelectionAndRegression(Module):
    def __init__(self, input_dim: int, feature_dim: int):
        super().__init__()
        self.selector = ConcreteFeatureSelection(input_dim=input_dim, output_dim=feature_dim)
        self.regressor = Linear(in_features=feature_dim, out_features=1)

    def forward(self, inputs):
        features, probabilities = self.selector(inputs)
        predictions = self.regressor(features)

        return predictions, probabilities


if __name__ == '__main__':
    samples = torch.randn((16, 300))  # Batch size, Num wavelengths
    model = SelectionAndRegression(input_dim=samples.shape[1], feature_dim=8)

    output_p, output_s = model(samples)
    print(output_p.shape)
    print(output_p)
