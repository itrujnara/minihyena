import math
from typing import Callable, Optional

from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fftconv(u, filter, bias, act=True):
    """
    Calculate long convolution using Fast Fourier Transform.
    """
    # fft size must be twice the sequence length to maintain causality
    fft_size = 2 * u.shape[-1] # u : [b, d, l]

    # apply FFT to the inputs
    u_fourier = torch.fft.rfft(u, n=fft_size) / fft_size
    filter_fourier = torch.fft.rfft(filter, n=fft_size) / fft_size

    # apply the convolution by multiplying and IFFT
    # we need to remove padding afterwards
    u_filter = u_fourier * filter_fourier
    y = torch.fft.irfft(u_filter, n=fft_size, norm="forward")[..., :u.shape[-1]]

    # apply bias
    out = y + u * bias.unsqueeze(1)

    # apply GeLU activation if needed
    if act:
        out = F.gelu(out)

    return out


class AsciiTokenizer():
    def tokenize(self, text):
        return list(np.frombuffer(text.encode(), dtype=np.uint8))
    
    def tokenize_batch(self, texts):
        return [self.tokenize(text) for text in texts]

    def detokenize(self, tokens):
        return "".join([chr(i) for i in tokens])
    

class OptimModule(nn.Module):
    """ Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters """

    def register(self, name, tensor, lr=None, wd=0.0):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {}
            if lr is not None:
                optim["lr"] = lr
            if wd is not None:
                optim["weight_decay"] = wd
            setattr(getattr(self, name), "_optim", optim)


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


class PositionalEmbedding(OptimModule):
    """Positional embedding based on complex exponentials.

    The combination of time with multiple cosine and sine frequencies is useful for this model.
    The time captures the uniqueness of each position.
    The sine and cosine waves at different frequencies capture different periodicities.
    """
    def __init__(self, emb_dim, seq_len, lr):
        super().__init__()

        self.seq_len = seq_len

        # time embedding
        t = torch.linspace(0, 1, self.seq_len)[None, :, None] # t : 1, L, 1

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


class ExponentialModulation(OptimModule):
    """Exponentially decay positions further from the current position.

    This increases the importance of close context over distant context.
    """
    def __init__(self, d_model, fast_decay_pct=0.3, slow_decay_pct=1.5, target=1e-2, lr=0.):
        super().__init__()
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
    

class RMSNorm(nn.Module):
    """Root Mean Square Normalization.

    This is a simple normalization that normalizes the input by the root mean square of the input.
    """
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.hidden_size = d_model
        self.scale = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # calculate the root mean square
        y = x / (x.norm(2, dim=-1, keepdim=True) * self.hidden_size ** (-1.0 / 2) + self.eps)
        # apply scale and bias
        y = y * self.scale + self.bias
        return y
    

class GatedMLP(nn.Module):
    """Gated MLP module."""
    def __init__(self, d_model, mlp_hidden_dim):
        super().__init__()
        self.l1 = nn.Linear(d_model, mlp_hidden_dim, bias=False)
        self.l2 = nn.Linear(d_model, mlp_hidden_dim, bias=False)
        self.l3 = nn.Linear(mlp_hidden_dim, d_model, bias=False)
        self.act = F.silu

    def forward(self, x):
        z1 = self.l1(x)
        z2 = self.l2(x)
        y = self.l3(self.act(z1) * z2)
        return y


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

        # activation
        act = Sin(dim=mlp_order, base_freq=act_freq)

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
            self.mlp.append(nn.Linear(mlp_order, mlp_order))
            self.mlp.append(act)
        # convert back to embedding dimension
        self.mlp.append(nn.Linear(mlp_order, d_model))

        # exponential modulation
        self.modulation = ExponentialModulation(d_model)

    def filter(self, L):
        # get the positional embedding
        z, t = self.pos_emb(L)
        # apply the MLP
        h = self.mlp(z)
        # apply exponential modulation
        h = self.modulation(t, h)
        return h

    def forward(self, x, L, conv_filter=None, bias=None, *args, **kwargs):
        # generate the filter with MLP if not provided
        if conv_filter is None:
            conv_filter = self.filter(L)

        # use default bias if not provided
        if bias is None:
            bias = self.bias
        
        # apply convolution
        y = fftconv(x, conv_filter, bias, act=False)

        return y


class HyenaOperator(nn.Module):
    """The Hyena operator as defined in the paper."""
    def __init__(
            self,
            d_model,
            l_max,
            order=2,
            filter_order=64,
            dropout=0.0,
            filter_dropout=0.0,
            **kwargs
        ):
        super().__init__()

        self.d_model = d_model
        self.l_max = l_max
        self.order = order
        inner_width = d_model * (order + 1)

        self.dropout = nn.Dropout(dropout)
        self.pre_norm = RMSNorm(d_model)
        self.post_norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, inner_width)
        self.out_proj = nn.Linear(d_model, d_model)

        self.short_filter = nn.Conv1d(
            inner_width,
            inner_width,
            3, # filter size
            padding=2,
            groups=inner_width
        )

        self.long_filter = HyenaFilter(
            d_model * (order - 1),
            mlp_order=filter_order,
            seq_len=l_max,
            dropout=filter_dropout,
            **kwargs
        )

    def forward(self, u, *args, **kwargs):
        L = u.shape[-2] # u : b l d
        l_filter = min(L, self.l_max)
        
        # normalize input
        z = self.pre_norm(u)
        # input projection
        z = self.in_proj(u)
        # reorder dimensions for short filter
        z = rearrange(z, "b l d -> b d l")
        
        # apply short filter
        uc = self.short_filter(z)[...,:l_filter] # uc: b inner_width l_filter
        # split into chunks of size d_model
        *x, v = uc.split(self.d_model, dim=1) # x_i: b d_model l_filter

        # generate long convolution filter
        conv_filter = self.long_filter.filter(l_filter)[0]
        # tweak dimensions for convolution
        conv_filter = rearrange(conv_filter, "l (o d) -> o d l", o=self.order - 1)
        bias = rearrange(self.long_filter.bias, "(o d) -> o d", o=self.order - 1)

        # apply hyena recursively
        for o, x_i in enumerate(reversed(x[1:])):
            # Hadamard product
            v = self.dropout(v * x_i)
            # convolution
            v = self.long_filter(v, l_filter, conv_filter=conv_filter[o], bias = bias[o])

        # final Hadamard product
        y = rearrange(v * x[0], "b d l -> b l d")

        # project back to input space
        y = self.out_proj(y) + u # residual connection
        # normalize output
        y = self.post_norm(y)

        return y
    

class AttentionBlock(nn.Module):
    """The attention block as defined in the paper."""
    def __init__(self, d_model, **kwargs):
        super().__init__()

        self.pre_norm = RMSNorm(d_model)
        self.post_norm = RMSNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, num_heads=8, **kwargs)
        self.mlp = GatedMLP(d_model, d_model * 2)

    def forward(self, x):
        u = self.pre_norm(x)
        y, _ = self.mha(u, u, u)
        y = y + u
        y = self.post_norm(y)
        y = self.mlp(y) + y

        return y


class HyenaBlock(nn.Module):
    """The Hyena block as defined in the paper."""
    def __init__(self, d_model, l_max, order=2, **kwargs):
        super().__init__()

        self.operator = HyenaOperator(d_model, l_max, order, **kwargs)
        self.activation = F.gelu

    def forward(self, x):
        y = self.operator(x)
        y = self.activation(y)

        return y
    

class MiniHyena(nn.Module):
    """A simplified version of StripedHyena."""
    def __init__(self, d_model, l_max, vocab_size, blocks):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.norm = RMSNorm(d_model)

        self.blocks = nn.Sequential()

        for i in blocks:
            if i == 'a':
                self.blocks.append(AttentionBlock(d_model))
            elif i == 'h':
                self.blocks.append(HyenaBlock(d_model, l_max))
            else:
                raise ValueError(f"Invalid block type: {i}")
            
    def forward(self, x):
        y = self.embedding(x)

        y = self.blocks(y)

        y = self.norm(y)

        # unembed
        y = y @ self.embedding.weight.T

        return y
    
    def compute_loss(self, output: torch.Tensor, target: torch.Tensor, loss_fn: Callable):
        output_reshaped = rearrange(output, 'b L d -> b d L')
        loss = loss_fn(output_reshaped, target)
        return loss
    
    def batch(
            self,
            x: dict,
            y: dict,
            loss_fn: Callable,
            optimizer: Optional[Callable] = None,
    ):
        tokens = x["input"]
        in_tokens = tokens[:, :-1]
        out_tokens = tokens[:, 1:]
        output = self(in_tokens)
        loss = self.compute_loss(output, out_tokens, loss_fn)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss, output

if __name__ == "__main__":
    # create the input
    sentence = "According to all known laws of aviation"
    print(sentence)

    tokenizer = AsciiTokenizer()
    tokens = tokenizer.tokenize(sentence)
    tokens = torch.tensor(tokens).repeat(10, 1).to(torch.int64)
    print(tokens.shape)

    # create the model
    model = MiniHyena(d_model=512, l_max=128, vocab_size=512, blocks="ah")

    output = model(tokens)
    print(output.shape)
