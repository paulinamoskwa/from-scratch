import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, masked: bool = False):
        super(SelfAttention, self).__init__()
        self.masked = masked

    def forward(self, q: torch.tensor, k: torch.tensor, v: torch.tensor) -> torch.tensor:
        d_k = k.size(-1)
        scores = torch.matmul(q, k.transpose(1, 2)) / (d_k ** 0.5)
      
        if self.masked:
            mask = torch.triu(torch.ones_like(scores), diagonal=1)
            scores = scores.masked_fill(mask == 1, float('-inf'))
        
        norm_scores = F.softmax(scores, -1)
        attention = torch.matmul(norm_scores, v)
        return attention
