from torch import nn
import torch

class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encodings for sequence data. Implemented as a torch.nn layer.


    Attributes:
        encoding (torch.Tensor): A precomputed matrix of positional encodings with shape
                                 (1, max_len, d_model). The matrix is computed using
                                 sinusoidal functions.
    """

    def __init__(self, d_model: int, seq_len: int):
        """
        Initializes the PositionalEncoding class.

        Args:
            d_model (int): The dimensionality of the input embeddings.
            seq_len (int): The length of the sequence.

        Behavior:
            - Precomputes a positional encoding matrix of shape (1, seq_len, d_model).
            - Uses sinusoidal functions to encode positions:
                - For even indices in the model dimension (2i):
                  PE(pos, 2i) = sin(pos / (10000^(2i/d_model)))
                - For odd indices (2i+1):
                  PE(pos, 2i+1) = cos(pos / (10000^(2i/d_model)))
        """
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        """
        Adds positional encodings to the input embeddings.

        Args:
            x (torch.Tensor): Input embeddings of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The input embeddings augmented with positional encodings.
                          The shape is the same as the input: (batch_size, seq_len, d_model).
       """
        return x + self.encoding[:, :x.size(1), :].to(x.device)