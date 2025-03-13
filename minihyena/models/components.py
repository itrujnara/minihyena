import math

import torch
from torch import nn

class Sin(nn.Module):
    """
    A module that applies a sine function with a trainable frequency to a tensor.
    """
    def __init__(self, dim, base_freq=1, trainable=True):
        super().__init__()
        # fix frequency to the base value or register as parameter
        self.freq = nn.Parameter(base_freq * torch.ones(1, dim)) if trainable else base_freq * torch.ones(1, dim)

    def forward(self, x):
        # multiplication by a number >1 squishes the sine
        return torch.sin(self.freq * x)


class PositionalEmbedding(nn.Module):
    """Positional embedding based on complex exponentials.

    The combination of time with multiple cosine and sine frequencies is useful for this model.
    The time captures the uniqueness of each position.
    The sine and cosine waves at different frequencies capture different periodicities.
    """
    def __init__(self, emb_dim, seq_len, lr):
        super().__init__()

        self.seq_len = seq_len

        # time embedding
        t = torch.linspace(0, 1, self.seq_len)[None, :, None] # t : [1, L, 1]

        # frequency embedding
        # there will be an equal number of cosine and sine components
        # how many?
        if emb_dim > 1: # don't calculate anything if emb_dim is 1
            bands = (emb_dim - 1) // 2
        # this will give us progress percentages...
        t_long = torch.linspace(0, self.seq_len - 1, self.seq_len)[None, :, None]
        # ...and this will give us evenly distributed angles 
        w = 2 * math.pi * t_long / seq_len
        # this will generate as many cosines and sines as required
        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        # we calculate the sines and cosines
        z = torch.exp(-1j * f * w) # Euler: e^-ia = cos a - i sin a
        # we convert the complex numbers to separate cos and sin vectors
        z = torch.cat([t, z.real, z.imag], dim=-1)
        # we register the embedding vectors
        self.register("z", z, lr=lr)
        self.register("t", t, lr=0.)

    def forward(self, L):
        # just return the embeddings
        return self.z[:, :L], self.t[:, :L]


class ExponentialModulation(nn.Module):
    """Exponentially decay positions further from the current position.

    This increases the importance of close context over distant context.
    """
    def __init__(self, d_model, fast_decay_pct=0.3, slow_decay_pct=1.5, target=1e-2, lr=0.):
        # exponent for the nearest position
        max_decay = math.log(target) / fast_decay_pct
        # exponent for the furthest position
        min_decay = math.log(target) / slow_decay_pct
        # everything in between
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        # register the exponents as a parameter
        self.register("deltas", deltas, lr=lr)

    def forward(self, t, x):
        decay = torch.exp(-t * self.deltas.abs())
        x = x * decay
        return x


class HyenaFilter(nn.Module):
    """The main Hyena filter.

    The main operation is an FFT convolution.
    The filter for the convolution can be generated with positional embedding and MLP.  
    """
    def __init__(
        self,
        d_model,
        emb_dim=3,
        mlp_order=16, # neurons per hidden layer
        mlp_hidden_layers=2,
        seq_len=1024,
        lr=1e-3,
        lr_emb=1e-5,
        dropout=0.,
        act_freq=1,
        weight_decay=0,
        normalized=False
    ):
        super().__init__()

        # basic variables
        self.d_model = d_model
        self.emb_dim = emb_dim
        self.seq_len = seq_len

        # trainable bias for FFT
        self.bias = nn.Parameter(torch.randn(self.d_model))
        # dropout for training
        self.dropout = nn.Dropout(dropout)

        # positional embedding
        self.pos_emb = PositionalEmbedding(emb_dim, seq_len, lr=lr_emb)

        # MLP construction
        # start with conversion from embedding to hidden dim
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_order),
            act
        )
        # apply the hidden layers
        for _ in range(mlp_hidden_layers):
            self.mlp.append(nn.Linear(order, order))
            self.mlp.append(act)
        # convert back to embedding dimension
        self.mlp.append(nn.Linear(order, emb_dim))

        # exponential modulation
        self.modulation = ExponentialModulation(d_model)

    def filter(self, L):
        # get the positional embedding
        z, t = self.pos_emb(L)
        # apply the MLP
        h = self.mlp(z)
        # apply exponential modulation
        h = self.modulation(t, h)


