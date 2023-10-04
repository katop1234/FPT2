
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
import pandas as pd
from torch.utils.checkpoint import checkpoint
import math
import constants

device = constants.device

class PrintableModule(nn.Module):
    def __init__(self):
        super(PrintableModule, self).__init__()
        self.print_frequency = 100
        self.counter = 0

    def forward(self, x):
        self.counter += 1
        return super().forward(x)

    def print_stats_bool(self):
        return self.counter % self.print_frequency == 0

    def print_similarity(self, a, b, block_name, depth):
        similarity = F.cosine_similarity(a, b, dim=-1)
        print(f"{block_name} similarity at depth {depth} in instance {type(self).__name__}: {similarity}")

    def print_stats(self, block_name, depth=0, a=None, b=None):
        if self.print_stats_bool():
            if a and b:
                self.print_similarity(a, b, block_name, depth)
            allocated_memory = round(torch.cuda.memory_allocated())
            cached_memory = round(torch.cuda.memory_cached())

            # Print memory stats
            print("Current memory allocated: ", allocated_memory, "Current memory cached: ", cached_memory)

# Alternative to positional embedding
class CategoryEmbedding(PrintableModule):
    def __init__(self, category_list, embed_dim):
        super().__init__()
        # Create a mapping from category strings to unique integers
        self.category_to_index = {category: i for i, category in enumerate(category_list)}
        
        self.embed = nn.Embedding(len(category_list), embed_dim).to(device)

    def forward(self, category):
        # Convert the category string to an integer index
        index = self.category_to_index[category]
        index = torch.tensor([index]).to(device)
        
        # Use this index to get the corresponding embedding
        return self.embed(index).squeeze(0)

    def __getitem__(self, category):
        return self.forward(category)

# Embeds any float or vector using an MLP
class ContinuousEmbeddingMLP(PrintableModule):
    def __init__(self, input_size, output_size, num_layers=3, bias=0., scale=1.):
        super().__init__()
        mlp_factor = 4

        self.norm = nn.LayerNorm(input_size)
        self.norm.weight.df.fill_(scale)
        self.norm.bias.df.fill_(bias)
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_size, mlp_factor * output_size))
        self.layers.append(nn.GELU())

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(mlp_factor * output_size, mlp_factor * output_size))
            self.layers.append(nn.GELU())

        # Output layer
        self.layers.append(nn.Linear(mlp_factor * output_size, output_size))

    def forward(self, x):
        x = self.norm(x)
        for layer in self.layers:
            x = layer(x)
        return x

class TickerEmbedding(PrintableModule):
    def __init__(self, ticker_list, embed_dim):
        super().__init__()
        self.ticker_to_index = {ticker: torch.tensor(i, dtype=torch.long).to(device) for i, ticker in enumerate(ticker_list)}
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(len(ticker_list), embed_dim).to(device)
        
        # Initialize all embeddings to be the same to make sure that the model doesn't overfit to variations in ticker embeddings
        self.embed.weight.data.fill_(0.02)  # or any other constant

    def forward(self, ticker):
        # Get the precomputed integer index tensor for the given ticker
        index_tensor = self.ticker_to_index[ticker]

        # Get the embedding for the index
        embedding = self.embed(index_tensor)

        return embedding


class Time2VecEmbedding(PrintableModule): # It seems "PrintableModule" is custom; use nn.Module if it's not available
    def __init__(self, embed_dim):
        super(Time2VecEmbedding, self).__init__() # Modified super() for clarity
        self.embed_dim = embed_dim
        
    def forward(self, time):
        # Compute the first half and second half of the tensor directly using slicing
        first_half = torch.sin(2 * math.pi * time[..., :self.embed_dim//2])
        
        second_half = time[..., self.embed_dim//2:]
        
        # Concatenating both parts
        combined = torch.cat((first_half, second_half), dim=-1)

        return combined

# TODO try ideas from mistral
# like swiglu activation (used in llama also)
# https://github.com/mistralai/mistral-src/blob/main/mistral/model.py

class Attention(nn.Module):
    '''
    Cross or Self Attention
    '''
    def __init__(
        self,
        dim,
        heads = 16,
    ):
        super().__init__()
        
        assert dim % heads == 0
        dim_head = dim // heads
        
        self.scale = dim_head ** -0.5
        self.heads = heads
        
        self.q_norm = nn.LayerNorm(dim_head)
        self.k_norm = nn.LayerNorm(dim_head)

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, dim, bias = True)
        self.to_k = nn.Linear(dim, dim, bias = True)
        self.to_v = nn.Linear(dim, dim, bias = True)
        self.to_out = nn.Linear(dim, dim, bias = False)

    def forward(
        self,
        x,
        attention_mask = None,
        context = None,
    ):
        h = self.heads
        x = self.norm(x)

        # print('x after norm', (torch.isnan(x).sum().item()), x)

        context = x if context is None else context

        qkv = (self.to_q(x), self.to_k(context), self.to_v(context))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # print('q', (torch.isnan(q).sum().item()), q)
        # print('k', (torch.isnan(k).sum().item()), k)
        # print('v', (torch.isnan(v).sum().item()), v)

        q, k = self.q_norm(q), self.k_norm(k) # Vit22B paper

        # print('q after norm', (torch.isnan(q).sum().item()), q)
        # print('k after norm', (torch.isnan(k).sum().item()), k)

        # print("min of q", torch.min(q), "max of q", torch.max(q))
        # print("min of k", torch.min(k), "max of k", torch.max(k))
        
        q = q * self.scale

        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)

        # print("sim", (torch.isnan(sim).sum().item()), sim)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(0)  # Add batch dimension
            attention_mask = attention_mask.unsqueeze(1).expand(-1, h, -1)  # Add heads dimension and expand to number of heads
            attention_mask = attention_mask.unsqueeze(2).expand(-1, h, -1, sim.size(-1))  # Add a dimension before seq_len

            sim.masked_fill_(~attention_mask, float('-inf'))

        # print("min of sim", torch.min(sim), "max of sim", torch.max(sim))

        attn = sim.softmax(dim = -1)

        # print("attn", (torch.isnan(attn).sum().item()), attn)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

        # print("out", (torch.isnan(out).sum().item()), out)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim, bias = True),
            nn.GELU(),
            nn.Linear(inner_dim, dim, bias = True)
        )

    def forward(self, x):
        x = self.norm(x)
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=16):
        super().__init__()

        self.attention = Attention(dim, heads)
        self.feed_forward = FeedForward(dim)

    def forward(self, x, attention_mask=None, context=None):
        x = self.attention(x, attention_mask, context) + x
        # print("x after cross attn", x, torch.min(x), torch.max(x))
        nan_count = torch.isnan(x).sum().item()
        # print(f"Number of NaN values: {nan_count}")
        x = self.feed_forward(x) + x
        # print("x after ff", x, torch.min(x), torch.max(x))
        nan_count = torch.isnan(x).sum().item()
        # print(f"Number of NaN values: {nan_count}")

        return x

class CheckpointedTransformerBlock(nn.Module):
    def __init__(self, dim, heads=16):
        super().__init__()

        self.cross_attention = Attention(dim, heads)
        self.feed_forward = FeedForward(dim)

    def forward(self, x, attention_mask=None, context=None):
        x = checkpoint(self.cross_attention, x, attention_mask, context) + x
        x = checkpoint(self.feed_forward, x) + x

        return x
