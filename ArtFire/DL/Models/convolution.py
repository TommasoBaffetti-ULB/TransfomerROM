import torch
import torch.nn as nn


class CNNblock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int = None,
        hidden_channels: list[int] | None = None,
        dim=2,
        kernel_sizes=3,
        strides=1,
        paddings=None,
        normalization: str | None = "Batch-Norm",
        activation: str = "gelu",
    ):
        super().__init__()

        assert dim in [1,2]
        assert output_channels is not None or hidden_channels is not None
        if hidden_channels is None:
            hidden_channels = []

        # Full channel progression
        channels = [input_channels] + hidden_channels
        if output_channels is not None:
            channels.append(output_channels)

        n_layers = len(channels) - 1

        # Expand scalar params to lists
        kernel_sizes = self._expand_param(kernel_sizes, n_layers, "kernel_sizes")
        strides = self._expand_param(strides, n_layers, "strides")

        if paddings is None:
            paddings = [k // 2 for k in kernel_sizes]
        else:
            paddings = self._expand_param(paddings, n_layers, "paddings")

        # Activation dictionary
        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "elu": nn.ELU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU,
            "silu": nn.SiLU,
        }

        if activation not in activations:
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Choose from {list(activations.keys())}."
            )

        if normalization not in ["Batch-Norm", "Layer-Norm", "Instance-Norm", "Group-Norm", None]:
            raise ValueError(
                "Unsupported normalization. Choose from "
                "['Batch-Norm', 'Layer-Norm', 'Instance-Norm', 'Group-Norm', None]"
            )

        activation_fn = activations[activation]
        layers = []

        for i in range(n_layers):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            k = kernel_sizes[i]
            s = strides[i]
            p = paddings[i]

            layers.append(
                self._make_conv(
                    dim=dim,
                    in_ch=in_ch,
                    out_ch=out_ch,
                    k=k,
                    s=s,
                    p=p,
                    bias=(normalization is None), ## batchnorm is equivalent to learn biases
                )
            )

            if normalization == "Batch-Norm":
                layers.append(nn.BatchNorm2d(out_ch))

            elif normalization == "Instance-Norm":
                layers.append(nn.InstanceNorm2d(out_ch, affine=True))

            elif normalization == "Group-Norm":
                num_groups = min(8, out_ch)
                while out_ch % num_groups != 0 and num_groups > 1:
                    num_groups -= 1
                layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=out_ch))

            layers.append(activation_fn())

        self.backbone = nn.Sequential(*layers)

    @staticmethod
    def _make_conv(dim, in_ch, out_ch, k, s, p, bias):
        if dim == 1:
            return nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias)
        return nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias)


    @staticmethod
    def _expand_param(param, n_layers: int, name: str):
        if isinstance(param, (int, tuple)):
            return [param] * n_layers
        if isinstance(param, list):
            if len(param) != n_layers:
                raise ValueError(f"{name} must have length {n_layers}, got {len(param)}")
            return param
        raise TypeError(f"{name} must be int, tuple, or list")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class TCNNblock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int = None,
        hidden_channels: list[int] | None = None,
        dim=2,
        kernel_sizes=3,
        strides=1,
        paddings=None,
        output_paddings=0,
        normalization: str | None = "Batch-Norm",
        activation: str = "gelu",
    ):
        super().__init__()

        assert dim in [1, 2]
        assert output_channels is not None or hidden_channels is not None
        if hidden_channels is None:
            hidden_channels = []

        channels = [input_channels] + hidden_channels
        if output_channels is not None:
            channels.append(output_channels)

        n_layers = len(channels) - 1

        kernel_sizes = self._expand_param(kernel_sizes, n_layers, "kernel_sizes")
        strides = self._expand_param(strides, n_layers, "strides")

        if paddings is None:
            paddings = [k // 2 if isinstance(k, int) else tuple(kk // 2 for kk in k) for k in kernel_sizes]
        else:
            paddings = self._expand_param(paddings, n_layers, "paddings")

        output_paddings = self._expand_param(output_paddings, n_layers, "output_paddings")

        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "elu": nn.ELU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU,
            "silu": nn.SiLU,
        }

        if activation not in activations:
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Choose from {list(activations.keys())}."
            )

        if normalization not in ["Batch-Norm", "Instance-Norm", "Group-Norm", None]:
            raise ValueError(
                "Unsupported normalization. Choose from "
                "['Batch-Norm', 'Instance-Norm', 'Group-Norm', None]"
            )

        activation_fn = activations[activation]
        layers = []

        for i in range(n_layers):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            k = kernel_sizes[i]
            s = strides[i]
            p = paddings[i]
            op = output_paddings[i]

            layers.append(
                self._make_conv_transpose(
                    dim=dim,
                    in_ch=in_ch,
                    out_ch=out_ch,
                    k=k,
                    s=s,
                    p=p,
                    op=op,
                    bias=(normalization is None),
                )
            )

            if normalization == "Batch-Norm":
                if dim == 1:
                    layers.append(nn.BatchNorm1d(out_ch))
                else:
                    layers.append(nn.BatchNorm2d(out_ch))

            elif normalization == "Instance-Norm":
                if dim == 1:
                    layers.append(nn.InstanceNorm1d(out_ch, affine=True))
                else:
                    layers.append(nn.InstanceNorm2d(out_ch, affine=True))

            elif normalization == "Group-Norm":
                num_groups = min(8, out_ch)
                while out_ch % num_groups != 0 and num_groups > 1:
                    num_groups -= 1
                layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=out_ch))

            layers.append(activation_fn())

        self.backbone = nn.Sequential(*layers)

    @staticmethod
    def _make_conv_transpose(dim, in_ch, out_ch, k, s, p, op, bias):
        if dim == 1:
            return nn.ConvTranspose1d(
                in_ch, out_ch, kernel_size=k, stride=s, padding=p,
                output_padding=op, bias=bias
            )
        return nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=k, stride=s, padding=p,
            output_padding=op, bias=bias
        )

    @staticmethod
    def _expand_param(param, n_layers: int, name: str):
        if isinstance(param, (int, tuple)):
            return [param] * n_layers
        if isinstance(param, list):
            if len(param) != n_layers:
                raise ValueError(f"{name} must have length {n_layers}, got {len(param)}")
            return param
        raise TypeError(f"{name} must be int, tuple, or list")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)