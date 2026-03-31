from ArtFire.DL.Models.CAE import CAE
from ArtFire.DL.Models.Forecast import ARTransformerForecaster
import torch.nn as nn


class Artfire(nn.Module):
    def __init__(self, cae: CAE, forecast: ARTransformerForecaster):
        super().__init__()

        self.cae = cae
        self.forecast = forecast

    def forward(self, x, horizon):
        x = self.cae.ConvEncoder(x)
        y = self.forecast(x, horizon=horizon)
        B, T, NT, D = y.shape
        y = y.view(B * T, NT, D)
        y = self.cae.ConvDecoder(y)
        return y.view(B, T, y.shape[1], y.shape[2], y.shape[3])
