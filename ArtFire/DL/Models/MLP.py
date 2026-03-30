import torch.nn as nn
import torch


class CustomMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim=None, hidden_layers=None, dropout_p= 0.0, normalization="Batch-Norm",
                 activation="leaky_relu", seed=42, initialization="kaiming uniform"):
        super().__init__()
        self.generator = torch.Generator().manual_seed(seed)
        assert (initialization in ["xavier uniform", " xavier normal", "kaiming uniform", "default"])
        assert (initialization != "kaiming uniform" or activation in ["relu", "leaky_relu"])
        assert (output_dim is not None or hidden_layers is not None)
        self.initialization = initialization
        layers = []

        # Choose the activation function
        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "elu": nn.ELU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation function '{activation}'. Choose from {list(activations.keys())}.")


        activation_fn = activations[activation]
        self.activation_fn = activation_fn
        self.activation=activation
        # Case 1: 0 Hidden Layers
        if not hidden_layers:
            layers.append(nn.Linear(input_dim, output_dim))

        # Case 2: 1 or More Hidden Layers
        else:
            # Input layer
            layers.append(nn.Linear(input_dim, hidden_layers[0]))
            if normalization=="Batch-Norm":
                layers.append(nn.BatchNorm1d(hidden_layers[0]))
            if normalization == "Layer-Norm":
                layers.append(nn.LayerNorm(hidden_layers[0]))
            layers.append(activation_fn())

            if dropout_p>0:
                layers.append(nn.Dropout(dropout_p))

            # Hidden layers
            for i in range(1, len(hidden_layers)):
                layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
                if normalization=="Batch-Norm":
                    layers.append(nn.BatchNorm1d(hidden_layers[i]))
                if normalization == "Layer-Norm":
                    layers.append(nn.LayerNorm(hidden_layers[i]))
                layers.append(activation_fn())
                if dropout_p>0:
                    layers.append(nn.Dropout(dropout_p))
            # Output layer (no activation or batch normalization)
            layers.append(nn.Linear(hidden_layers[-1], output_dim))

        self.backbone = nn.Sequential(*layers)

        if initialization != "default":
            self.initialize()


    def initialize(self):
        for l in range(len(self.backbone)):
            custom_init(self.backbone[l], generator=self.generator, init_type=self.initialization,
                        nonlinearity=self.activation)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def custom_init(m, generator=None, bias_range=0.005, init_type="uniform", nonlinearity=None):
    if isinstance(m, nn.Linear):
        if init_type == "xavier uniform":
            nn.init.xavier_uniform_(m.weight, generator=generator)  # Xavier uniform
        if init_type == "xavier normal":
            nn.init.xavier_normal_(m.weight, generator=generator)  # Xavier normal
        if init_type == "kaiming uniform":
            nn.init.kaiming_uniform_(m.weight, generator=generator, nonlinearity=nonlinearity)

        nn.init.constant_(m.bias, bias_range)

