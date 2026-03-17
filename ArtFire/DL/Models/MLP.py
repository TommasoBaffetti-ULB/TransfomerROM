import torch.nn as nn
import torch


class CustomRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layers=None, dropout=0.0, normalization="Batch-Norm",
                 activation="leaky_relu", seed=42, initialization="kaiming uniform"):
        """
        Custom MLP Regressor with configurable hidden layers, batch/Layer normalization, dropout, and activation functions.
        Args:
            input_dim (int): Dimension of the input state.
            output_dim (int): Number of actions (output Q-values).
            hidden_layers (list or None): List of hidden layer sizes (e.g., [64, 64]).
                                         Use `None` or an empty list for 0 hidden layers -> input layer - output layer.
            dropout (float): Dropout rate (default: 0.0).
            normalization (string): Whether to use batch normalization (default: Batch-Norm).
            activation (str): Activation function to use ("relu", "gelu", "elu", "tanh", "sigmoid", "leaky_relu" ).
        """
        super(CustomRegressor, self).__init__()
        self.generator = torch.Generator().manual_seed(seed)
        assert (initialization in ["xavier uniform", " xavier normal", "kaiming uniform", "default"])
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

        if normalization not in ["Batch-Norm", "Layer-Norm"]:
            raise ValueError(f"Unsupported normalization '{normalization}'. Choose from 'Batch-Norm' 'Layer-Norm'")
        activation_fn = activations[activation]
        self.activation = activation

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
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            # Hidden layers
            for i in range(1, len(hidden_layers)):
                layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
                if normalization=="Batch-Norm":
                    layers.append(nn.BatchNorm1d(hidden_layers[i]))
                if normalization == "Layer-Norm":
                    layers.append(nn.LayerNorm(hidden_layers[i]))
                layers.append(activation_fn())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

            # Output layer (no activation or batch normalization)
            layers.append(nn.Linear(hidden_layers[-1], output_dim))

        self.network = nn.Sequential(*layers)
        if initialization != "default":
            self.initialize()  # not called because I am not sure it is properly done

    def initialize(self):
        for l in range(len(self.network)):
            custom_init(self.network[l], generator=self.generator, init_type=self.initialization,
                        nonlinearity=self.activation)

    def forward(self, x):
        return self.network(x)




class CustomSoftmax(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layers=None, dropout=0.0, normalization="Batch-Norm",
                 activation="leaky_relu", seed=42, initialization="kaiming uniform"):
        """
        Custom MLP Regressor with configurable hidden layers, batch normalization, dropout, and activation functions.
        Args:
            input_dim (int): Dimension of the input state.
            output_dim (int): Number of actions (output Q-values).
            hidden_layers (list or None): List of hidden layer sizes (e.g., [64, 64]).
                                         Use `None` or an empty list for 0 hidden layers -> input layer - output layer.
            dropout (float): Dropout rate (default: 0.0).
            use_batch_norm (bool): Whether to use batch normalization (default: False).
            activation (str): Activation function to use ("relu", "gelu", "elu", "tanh", "sigmoid", "leaky_relu" ).
        """
        super(CustomSoftmax, self).__init__()
        self.generator = torch.Generator().manual_seed(seed)
        assert (initialization in ["xavier uniform", " xavier normal", "kaiming uniform", "default"])
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
        self.activation = activation

        # Case 1: 0 Hidden Layers
        if not hidden_layers:
            layers.append(nn.Linear(input_dim, output_dim))

        # Case 2: 1 or More Hidden Layers
        else:
            # Input layer
            layers.append(nn.Linear(input_dim, hidden_layers[0]))
            if normalization== "Batch-Norm":
                layers.append(nn.BatchNorm1d(hidden_layers[0]))
            if normalization == "Layer-Norm":
                layers.append(nn.LayerNorm(hidden_layers[0]))
            layers.append(activation_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            # Hidden layers
            for i in range(1, len(hidden_layers)):
                layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
                if normalization=="Batch-Norm":
                    layers.append(nn.BatchNorm1d(hidden_layers[i]))
                if normalization == "Layer-Norm":
                    layers.append(nn.LayerNorm(hidden_layers[i]))
                layers.append(activation_fn())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

            # Output layer (no activation or batch normalization)
            layers.append(nn.Linear(hidden_layers[-1], output_dim))
            layers.append(nn.Softmax(dim=1))

        self.network = nn.Sequential(*layers)
        if initialization!="default":
            self.initialize()   # not called because I am not sure it is properly done

    def initialize(self):
        for l in range(len(self.network)):
            custom_init(self.network[l], generator=self.generator, init_type=self.initialization,
                        nonlinearity=self.activation)

    def forward(self, x):
        return self.network(x)





def custom_init(m, generator=None, init_type="uniform", nonlinearity=None):
    if isinstance(m, nn.Linear):
        if init_type == "xavier uniform":
            nn.init.xavier_uniform_(m.weight, generator=generator)  # Xavier uniform
        if init_type == "xavier normal":
            nn.init.xavier_normal_(m.weight, generator=generator)  # Xavier normal
        if init_type == "kaiming uniform":
            nn.init.kaiming_uniform_(m.weight, generator=generator, nonlinearity=nonlinearity)
