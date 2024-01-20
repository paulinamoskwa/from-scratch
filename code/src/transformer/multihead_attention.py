import torch
import torch.nn as nn
from self_attention import SelfAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int = 6, d_model: int = 512, masked: bool = False):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.attention_mechanism = SelfAttention(masked=masked)
        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)
        self.w_o = nn.Linear(self.d_model * self.n_heads, self.d_model)

    def forward(self, q: torch.tensor, k: torch.tensor, v: torch.tensor) -> torch.tensor:
        assert (q.size(2) == self.d_model), f"Error: the embedding space is not {self.d_model}-dim"
        concatenated_attentions = torch.empty(0)
        for i in range(self.n_heads):
            proj_q = self.w_q(q)
            proj_k = self.w_k(k)
            proj_v = self.w_v(v)
            attention = self.attention_mechanism(proj_q, proj_k, proj_v)
            concatenated_attentions = torch.cat([concatenated_attentions, attention], dim=-1)
        return self.w_o(concatenated_attentions)
