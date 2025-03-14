import torch
from torch import nn
from torch.nn import functional as F

from components import HyenaOperator, RMSNorm, GatedMLP


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
            
    

if __name__ == "__main__":
    # make some input
    x = torch.randn(4, 1024, 512)
    print(f"Input shape: {x.shape}")

    # test the AttentionBlock
    attention = AttentionBlock(512)
    y = attention(x)
    print(f"Attention output shape: {y.shape}")

    # test the HyenaBlock
    hyena = HyenaBlock(512, 1024)
    z = hyena(x)
    print(f"Hyena output shape: {z.shape}")

    # test the MiniHyena
    tokens = torch.randint(0, 512, (4, 1024))
    model = MiniHyena(512, 1024, 512, "hah")
    y = model(tokens)
    print(f"Tokens shape: {tokens.shape}")
    print(f"MiniHyena output shape: {y.shape}")
    print(f"MiniHyena sample output: {y[0, -1, :]}")
