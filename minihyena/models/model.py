from typing import Callable, Optional

from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as F

from components import HyenaOperator, RMSNorm, GatedMLP
from tokenizer import AsciiTokenizer


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

    # run the model
    x = {"input": tokens}
    y = dict()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 10
    for epoch in range(n_epochs):
        loss, output = model.batch(x, y, loss_fn, optimizer)
        print(f"Epoch {epoch+1}/{n_epochs} - Loss: {loss.item()}")
    # print(output.shape)
    # print(output[0])
    # print(output[0].argmax(dim=-1))
    # print(tokenizer.detokenize(output[0].argmax(dim=-1)))
    test_sentence = "known laws of avi"
    test_tokens = tokenizer.tokenize(test_sentence)
    test_tokens = torch.tensor(test_tokens).unsqueeze(0).to(torch.int64)
    test_output = model(test_tokens)
    print(test_output.shape)
    print(test_output[0])
    print(test_output[0].argmax(dim=-1))
    print(tokenizer.detokenize(test_output[0].argmax(dim=-1)))
    
