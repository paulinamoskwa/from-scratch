import torch
import torch.nn as nn
from multihead_attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, n_heads: int = 6, d_model: int = 512):
        super(EncoderLayer, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model

        # Multi-head attention sub-layer
        self.multihead_attention = MultiHeadAttention(n_heads, d_model)
        self.norm1 = nn.LayerNorm(d_model)

        # Feed-forward sub-layer
        self.feedforward_1 = nn.Linear(d_model, d_model)
        self.relu = nn.ReLU()
        self.feedforward_2 = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # Multi-head attention sub-layer
        attention = self.multihead_attention(x, x, x)
        x = x + attention
        x = self.norm1(x)

        # Feed-forward sub-Layer
        feedforward_output = self.feedforward_1(x)
        feedforward_output = self.relu(feedforward_output)
        feedforward_output = self.feedforward_2(feedforward_output)
        x = x + feedforward_output
        x = self.norm2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, n_layers: int = 6, n_heads: int = 6, d_model: int = 512):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.encoder_layer = EncoderLayer(n_heads, d_model)

    def forward(self, x: torch.tensor) -> torch.tensor:
        for i in range(self.n_layers):
            x = self.encoder_layer(x)
        return x
