from torch import nn
import torch
import utils
from classes import (
    TransformerBlock, ContinuousUnembedding, Time2VecEmbedding, 
    TickerEmbedding, ContinuousEmbedding, CategoryEmbedding)
import numpy as np
import pandas as pd
import torch.nn.functional as F
import constants

device = constants.device

class FPT(nn.Module):
    def __init__(self,
                 embed_dim=1280,
                 depth=32,
                 batch_size=100,
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.batch_size = batch_size
        
        self.categories_lookup = dict()
        self.categories_lookup["floats"] = utils.get_floats_categories()
        self.categories_lookup["text"] = utils.get_text_categories()
        self.categories_lookup["datetime"] = utils.get_time2vec_categories()
        self.all_categories = self.categories_lookup["floats"] + self.categories_lookup["datetime"] # + self.categories_lookup["text"] 
        self.seq_len = len(self.all_categories) + 1 # +1 for the cls token
        
        # (Only ["Ticker"] for now)
        ticker_list = utils.get_ticker_list()

        # TODO change this from being hardcoded to 4
        self.cont_embeds = nn.ModuleList([ContinuousEmbedding(embed_dim, embed_dim) for _ in range(self.seq_len - 4 - 1)]) # 1 for cls
        self.time2vec_embeds = nn.ModuleList([Time2VecEmbedding(embed_dim) for _ in range(4)])

        ## Get categorical embeddings and mask tokens to add to input embedding
        self.cat_embeddings = torch.randn(self.seq_len - 1, self.embed_dim) * 0.02 # remove 1 for cls
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim)).to(device) * 0.02
        self.decoder_input_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.decoder_norm = nn.LayerNorm(self.embed_dim)
        
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                )
                for _ in range(depth)
            ]
        )
        
        self.predictor = ContinuousUnembedding(self.embed_dim, 1)

    
    def append_cls(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        return x


    def pad_x_for_embed_dim(self, tensor):
        padding_size = self.embed_dim - tensor.size(1)
        if padding_size > 0:
            tensor = F.pad(tensor, (0, padding_size))
        return tensor
    
    def forward_decoder(self, x, attention_mask=None):
        
        x = self.pad_x_for_embed_dim(x)
        x = x.unsqueeze(0)

        # Separate tokens
        out = []
        
        # Apply ContinuousEmbedding to the first (seq_len - 4) tokens
        # TODO see if this can be parallelized
        for i in range(len(self.cont_embeds)):
            out.append(self.cont_embeds[i](x[:, i, :]).unsqueeze(1))
        
        # Apply Time2VecEmbedding to the last 4 tokens
        for i in range(len(self.time2vec_embeds)):
            out.append(self.time2vec_embeds[i](x[:, -4 + i, :]).unsqueeze(1)) # TODO don't hardcode 4
            # TODO have year be a continuous embedding since its not periodic
        
        # Concatenate results across the sequence length dimension
        x = torch.cat(out, dim=1)

        x += self.cat_embeddings
        
        # TODO properly add batches across GPUs
        
        x = self.decoder_input_proj(x)
        x = self.append_cls(x)
        
        depth = 0
        for blk in self.decoder_blocks:
            x = blk(x, attention_mask)
            # TODO this throws nans after 1-2 layers!!!
            # TODO change the initialization for the weights
            depth += 1
            print("cls token at depth", depth, x[:, 0, :])
        x = self.decoder_norm(x)
        cls_token = x[:, 0, :]
        return cls_token
    
    def forward_loss(self, cls_token, gt):
        pred = self.predictor(cls_token)
        
        print("types", type(gt), type(pred))
        print("actuals", gt, pred)
        print("shapes", gt.shape, pred.shape)
        
        loss = utils.mean_squared_error(gt, pred)
        return loss
    
    def forward(self, x, attention_mask, gt):

        x_np = x.numpy()
        # Save numpy array to a text file
        np.savetxt("tensor_data.txt", x_np, fmt="%s", delimiter=",")

        # TODO also change the Time2Vec embedding class to never worry abt modulus other than 1

        # TODO for datetime vectors, just copy the value over repeatedly, don't give it a bunch of 0s

        # TODO consider making a single token for datetime values

        cls_token = self.forward_decoder(x, attention_mask)
        loss = self.forward_loss(cls_token, gt)
        return loss
        