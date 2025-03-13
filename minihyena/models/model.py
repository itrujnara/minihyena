import torch
from torch import nn
from torch.nn import functional as F

from components import HyenaOperator


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
    

if __name__ == "__main__":
    # test the HyenaBlock
    block = HyenaBlock(512, 1024, order=2)
    x = torch.randn(4, 1024, 512)

    y = block(x)
    print(y.shape)
