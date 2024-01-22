import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import Encoder
from decoder import Decoder
from positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(self, target_vocab_size: int, n_layers: int = 6, n_heads: int = 6, d_model: int = 512):
        super(Transformer, self).__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.target_vocab_size = target_vocab_size

        self.positional_encoding = PositionalEncoding()
        self.encoder = Encoder(self.n_layers, self.n_heads, self.d_model)
        self.decoder = Decoder(self.n_layers, self.n_heads, self.d_model)
        self.positional_encoding = PositionalEncoding()

        # Linear layer for output projection
        self.output_projection = nn.Linear(self.d_model, self.target_vocab_size)

    def forward(self, source: torch.tensor, target: torch.tensor) -> torch.tensor:
        source = source + self.positional_encoding(source)
        target = target + self.positional_encoding(target)

        encoder_output = self.encoder(source)
        decoder_output = self.decoder(target, encoder_output)

        # Project the decoder output to the target's vocabulary space
        output_logits = self.output_projection(decoder_output)

        return F.softmax(output_logits, dim=-1)
