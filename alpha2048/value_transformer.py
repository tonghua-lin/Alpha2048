"""Small encoder for exponent-encoded 2048 afterstates; outputs raw teacher scores."""
import torch
from torch import nn


class ValueTransformer(nn.Module):
    def __init__(self, vocab_size=32, width=64, layers=4, heads=4, feedforward=256,
                 target_mean=0.0, target_std=1.0):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, width=width, layers=layers,
                           heads=heads, feedforward=feedforward)
        self.tiles = nn.Embedding(vocab_size, width)
        self.rows = nn.Embedding(4, width)
        self.cols = nn.Embedding(4, width)
        self.register_buffer('row_ids', torch.arange(4).repeat_interleave(4))
        self.register_buffer('col_ids', torch.arange(4).repeat(4))
        self.register_buffer('target_mean', torch.tensor(float(target_mean)))
        self.register_buffer('target_std', torch.tensor(float(target_std)))
        # Construct separately so layers do not start as identical deep copies.
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(width, heads, feedforward, dropout=0.0,
                                       activation='gelu', batch_first=True, norm_first=True)
            for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, boards):
        x = self.tiles(boards.reshape(-1, 16).long())
        x = x + self.rows(self.row_ids) + self.cols(self.col_ids)
        for block in self.blocks:
            x = block(x)
        value = self.head(self.norm(x).mean(dim=1)).squeeze(-1)
        return value * self.target_std + self.target_mean


def d4_indices():
    """Eight spatial symmetries; tile values and target scores stay unchanged."""
    grid = torch.arange(16).reshape(4, 4)
    return torch.stack([torch.rot90(g, k, (0, 1)).flatten()
                        for g in (grid, grid.flip(1)) for k in range(4)])


def load_value_model(path, device='cpu'):
    # Keep existing imports working while accepting both architectures.
    from .value_models import load_value_model as load
    return load(path, device)
