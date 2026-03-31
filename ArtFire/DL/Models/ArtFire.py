from ArtFire.DL.Models.CAE import CAE
from ArtFire.DL.Models.Forecast import ARTransformerForecaster
import torch.nn as nn


class Artfire(nn.Module):
    def __init__(self, cae: CAE, forecast: ARTransformerForecaster):
        super().__init__()

        self.cae = cae
        self.forecast = forecast

    def forward(self, x):
        x = self.cae.ConvEncoder(x)
        y = self.forecast(x)
        return self.cae.ConvDecoder(y)
