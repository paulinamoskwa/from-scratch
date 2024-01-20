import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def forward(self, x: torch.tensor) -> torch.tensor:
        pe = torch.ones_like(x, dtype=torch.float32)
        d_model = pe.size(-1)
        len_seq = pe.size(-2)

        for pos in range(len_seq):
            for i in range(d_model // 2):
                exp_term = 2 * i / d_model
                div_term = 10000 ** exp_term
                pe[:, pos, 2 * i] = torch.sin(torch.tensor(pos / div_term))
                pe[:, pos, 2 * i + 1] = torch.cos(torch.tensor(pos / div_term))

            # Modify the last element explicitly if d_model is odd
            if d_model % 2 != 0:
                i = d_model // 2
                exp_term = 2 * i / d_model
                div_term = 10000 ** exp_term
                pe[:, pos, 2 * i] = torch.sin(torch.tensor(pos / div_term))

        return pe
