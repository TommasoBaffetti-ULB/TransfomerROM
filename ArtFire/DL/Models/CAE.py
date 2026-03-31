from ArtFire.DL.Models.convolution import CNNblock, TCNNblock
import torch.nn as nn


class CAE(nn.Module):
    def __init__(self, ConvEncoder, ConvDecoder):
        super().__init__()
        self.ConvEncoder = ConvEncoder
        self.ConvDecoder = ConvDecoder

    def forward(self, x):
        x = self.ConvEncoder(x)
        return self.ConvDecoder(x)


class ConvEncoder(nn.Module):
    def __init__(self, config_spatial_conv, config_temporal_conv):
        super().__init__()
        self.sp_conv = CNNblock(**config_spatial_conv)
        self.t_conv = CNNblock(**config_temporal_conv)

    def forward(self, x):
        x = self.sp_conv(x)
        B, C, H, W = x.shape
        x = x.permute((0, 2, 1, 3))  # -> B,H,C,W
        x = x.unsqueeze(2)  # -> B,H,1,C,W
        x = x.reshape(B * H, 1, C, W)  # -> B*H,1,C,W
        x = self.t_conv(x)
        x = x.view(B, H, *x.shape[1:])
        return x.flatten(start_dim=2)  # .reshape((B,H,-1))


class ConvDecoder(nn.Module):
    def __init__(self, config_spatial_tconv, config_temporal_tconv, no_flatten_dim):
        super().__init__()
        self.sp_tconv = TCNNblock(**config_spatial_tconv)
        self.t_tconv = TCNNblock(**config_temporal_tconv)
        self.C2, self.H2, self.W2 = no_flatten_dim

    def forward(self, x):
        B, H = x.shape[:2]
        x = x.view(B * H, self.C2, self.H2, self.W2)
        x = self.t_tconv(x)
        x = x.view(B, H, *x.shape[1:])
        x = x.squeeze(dim=2)
        x = x.permute(0, 2, 1, 3)
        return self.sp_tconv(x)
