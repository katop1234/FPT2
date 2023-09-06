
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
import pandas as pd
from torch.utils.checkpoint import checkpoint
import math

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
        
        self.embed = nn.Embedding(len(category_list), embed_dim).cuda()

    def forward(self, category):
        # Convert the category string to an integer index
        index = self.category_to_index[category]
        index = torch.tensor([index]).cuda()
        
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
    
# simple linear projection from input_size to output_size
# previous MLP embedder might be overkill
class ContinuousEmbedding(PrintableModule):
    def __init__(self, input_size, output_size, bias=0., scale=1.):
        super().__init__()
        
        self.norm = nn.LayerNorm(input_size)
        self.norm.weight.df.fill_(scale)
        self.norm.bias.df.fill_(bias)
        
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        x = self.norm(x) # Technically redundant but useful for large values like year which is 2000 and we need it to be 0->1
        return self.linear(x)

class TickerEmbedding(nn.Module):
    def __init__(self, ticker_list, embed_dim):
        super().__init__()
        self.ticker_to_index = {ticker: torch.tensor(i, dtype=torch.long).cuda() for i, ticker in enumerate(ticker_list)}
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(len(ticker_list), embed_dim).cuda()
        
        # Initialize all embeddings to be the same to make sure that the model doesn't overfit to variations in ticker embeddings
        self.embed.weight.data.fill_(0.02)  # or any other constant

    def forward(self, ticker):
        # Get the precomputed integer index tensor for the given ticker
        index_tensor = self.ticker_to_index[ticker]

        # Get the embedding for the index
        embedding = self.embed(index_tensor)

        return embedding

    
class Time2VecEmbedding(PrintableModule):
    def __init__(self, modulus, embed_dim):
        super().__init__()
        self.modulus = modulus
        self.periodic_linear = nn.Linear(1, embed_dim // 2)
        self.linear = nn.Linear(1, embed_dim // 2)
        
    def forward(self, time):
        time = time % self.modulus
        omega = 2 * math.pi / self.modulus

        # Apply the periodic and linear transformations
        periodic_transform = torch.sin(omega * self.periodic_linear(time))
        linear_transform = self.linear(time)
        
        # Concatenate along the last dimension to create the final embedding
        embedding = torch.cat([periodic_transform, linear_transform], dim=-1)

        return embedding

# Maps the output of the decoder back to the original value
class ContinuousUnembedding(ContinuousEmbedding):
    pass

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
        
        self.q_norm = nn.LayerNorm(dim)
        self.k_norm = nn.LayerNorm(dim)

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, dim, bias = True)
        self.to_k = nn.Linear(dim, dim, bias = True)
        self.to_v = nn.Linear(dim, dim, bias = True)
        self.to_out = nn.Linear(dim, dim, bias = False)

    def forward(
        self,
        x,
        context = None,
        attention_mask = None,
    ):
        h = self.heads
        x = self.norm(x)

        context = x if context is None else context

        qkv = (self.to_q(x), self.to_k(context), self.to_v(context))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        
        q, k = self.q_norm(q), self.k_norm(k) # Vit22B paper
        q = q * self.scale

        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)

        if attention_mask is not None:
            sim.masked_fill_(~attention_mask.unsqueeze(1).unsqueeze(2), float('-inf'))  # Fill masked positions with a large negative number

        attn = sim.softmax(dim = -1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
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

        self.cross_attention = Attention(dim, heads)
        self.feed_forward = FeedForward(dim)

    def forward(self, x, context=None, attention_mask=None):
        x = self.cross_attention(x, context, attention_mask) + x
        x = self.feed_forward(x) + x

        return x

class CheckpointedTransformerBlock(nn.Module):
    def __init__(self, dim, heads=16):
        super().__init__()

        self.cross_attention = Attention(dim, heads)
        self.feed_forward = FeedForward(dim)

    def forward(self, x, context=None):
        x = checkpoint(self.cross_attention, x, context) + x
        x = checkpoint(self.feed_forward, x) + x

        return x
