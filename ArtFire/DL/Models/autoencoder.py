class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encodings for sequence data. Implemented as a torch.nn layer.

    Positional encodings are added to input embeddings to provide information about the
    positions of tokens in the sequence. They are commonly used in transformer-based models.

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


class TransformerAutoencoder(nn.Module):
    """
    A Transformer-based Autoencoder for sequence compression and reconstruction.

    This model compresses an input sequence into a lower-dimensional representation and reconstructs it
    using a Transformer-based architecture. It includes positional encoding, transformer encoder-decoder
    layers, and compression/decompression FeedForward linear layers.

    Attributes:
        input_dim (int): The number of features in each input sequence element.
        seq_len (int): The length of the input sequences.
        d_model (int): The dimensionality of the transformer model (embedding size).
        input_projection (nn.Linear): A linear layer to project input features to d_model.
        positional_encoding (PositionalEncoding): A positional encoding module to add position information.
        compression (nn.Linear): A linear layer to compress the encoder output into a single feature.
        decompression (nn.Linear): A linear layer to expand the compressed representation back to d_model.
        transformer_encoder (nn.TransformerEncoder): A stack of transformer encoder layers.
        transformer_decoder (nn.TransformerDecoder): A stack of transformer decoder layers.
        output_projection (nn.Linear): A linear layer to project the output of the decoder back to input_dim.
    """

    def __init__(self, input_dim=6, seq_len=30, d_model=64, nhead=4, num_layers=2):
        """
        Initializes the TransformerAutoencoder.

        Args:
            input_dim (int): The number of features in the input sequence. default 6 p,q,v bus1 and p,q,v bus2 (maybe not in this order p,q,v)
            seq_len (int): The length of the input sequences.
            d_model (int): The dimensionality of the transformer model.
            nhead (int): The number of attention heads in the transformer.
            num_layers (int): The number of encoder and decoder layers.
        """
        super(TransformerAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model

        # Input projection to d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, seq_len=seq_len)

        # Transformer Encoder-Decoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.3, batch_first=True)
        self.compression = nn.Linear(d_model, 1)

        # Decompression layer to expand back to d_model
        self.decompression = nn.Linear(1, d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=0.3, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection back to input_dim
        self.output_projection = nn.Linear(d_model, input_dim)

    def forward(self, x):
        """
        Forward pass of the autoencoder.

        Args:
            x (torch.Tensor): Input sequence of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Reconstructed sequence of shape (batch_size, seq_len, input_dim).
        """
        # Project input to d_model
        x = self.input_projection(x)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Encode
        encoded = self.transformer_encoder(x)

        compressed = self.compression(encoded)

        decompressed = self.decompression(compressed)

        # Decode (reconstruct)
        decoded = self.transformer_decoder(decompressed, decompressed)

        # Project back to input_dim
        reconstructed = self.output_projection(decoded)
        return reconstructed

    def compress(self, x):
        """
        Compresses the input sequence into a lower-dimensional representation.
        Apply only the encoding part

        Args:
            x (torch.Tensor): Input sequence of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Compressed sequence of shape (batch_size, seq_len, 1).
        """
        # Project input to d_model
        x = self.input_projection(x)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Encode
        encoded = self.transformer_encoder(x)

        compressed = self.compression(encoded)
        return compressed
