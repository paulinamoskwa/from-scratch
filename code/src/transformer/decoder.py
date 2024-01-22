import torch
import torch.nn as nn
from multihead_attention import MultiHeadAttention


class DecoderLayer(nn.Module):
    def __init__(self, n_heads: int = 6, d_model: int = 512):
        super(DecoderLayer, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model

        # Masked multi-head attention sub-layer
        self.masked_multihead_attention = MultiHeadAttention(self.n_heads, self.d_model, masked=True)
        self.norm1 = nn.LayerNorm(self.d_model)

        # Multi-head attention sub-layer
        self.multihead_attention = MultiHeadAttention(self.n_heads, self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)

        # Feed-forward sub-layer
        self.feedforward_1 = nn.Linear(self.d_model, self.d_model)
        self.relu = nn.ReLU()
        self.feedforward_2 = nn.Linear(self.d_model, self.d_model)
        self.norm3 = nn.LayerNorm(self.d_model)

    def forward(self, x: torch.tensor, encoder_output: torch.tensor) -> torch.tensor:
        # Masked multi-head attention sub-layer
        masked_attention = self.masked_multihead_attention(x, x, x)
        x = x + masked_attention
        x = self.norm1(x)

        # Multi-head attention sub-layer with encoder output
        attention_output = self.multihead_attention(x, encoder_output, encoder_output)
        x = x + attention_output
        x = self.norm2(x)

        # Feed-forward sub-Layer
        feedforward_output = self.feedforward_1(x)
        feedforward_output = self.relu(feedforward_output)
        feedforward_output = self.feedforward_2(feedforward_output)
        x = x + feedforward_output
        x = self.norm3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, n_layers: int = 6, n_heads: int = 6, d_model: int = 512):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.decoder_layer = DecoderLayer(self.n_heads, self.d_model)

    def forward(self, x: torch.tensor, encoder_output: torch.tensor) -> torch.tensor:
        for i in range(self.n_layers):
            x = self.decoder_layer(x, encoder_output)
        return x
