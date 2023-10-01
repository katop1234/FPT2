from torch import nn
import torch
import utils
from classes import (
    TransformerBlock, ContinuousUnembedding, Time2VecEmbedding, 
    TickerEmbedding, LinearProjection, CategoryEmbedding)
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
        self.ticker_list = utils.get_ticker_list()

        # TODO change this from being hardcoded to 4
        self.continuous_embeddings = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(self.seq_len - 4 - 1)]) # 1 for cls
        self.time2vec_embeddings = nn.ModuleList([Time2VecEmbedding(embed_dim) for _ in range(4)])

        ## Get categorical embeddings and mask tokens to add to input embedding
        self.categorical_embeddings = torch.randn(self.seq_len - 1, self.embed_dim) * 0.02 # remove 1 for cls
        
        self.decoder_input_norm = nn.LayerNorm(self.embed_dim)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim)).to(device) * 0.02
        # self.decoder_input_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.decoder_output_norm = nn.LayerNorm(self.embed_dim)
        
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                )
                for _ in range(depth)
            ]
        )
        
        self.predictor = nn.Linear(self.embed_dim, 1)

        def initialize_weights(module):
            if isinstance(module, (nn.Linear)):
                module.weight.data.normal_(mean=0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

        self.apply(initialize_weights)
    
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

        nan_count = torch.isnan(x).sum().item()
        print(f"Number of NaN values in input: {nan_count}")
        if nan_count > 0:
            print("got", nan_count, "nans")
            exit()
        
        x = self.pad_x_for_embed_dim(x)
        x = x.unsqueeze(0)

        # Separate tokens
        out = []

        # Apply ContinuousEmbedding to the first (seq_len - 4) tokens
        # TODO see if this can be parallelized
        for i in range(len(self.continuous_embeddings)):
            embed_output = self.continuous_embeddings[i](x[:, i, :]).unsqueeze(1)
            nan_count = torch.isnan(embed_output).sum().item()
            if nan_count > 0:
                print(f"ContinuousEmbedding loop added {nan_count} NaNs at index {i}")
            out.append(embed_output)

        # Apply Time2VecEmbedding to the last 4 tokens
        for i in range(len(self.time2vec_embeddings)):
            embed_output = self.time2vec_embeddings[i](x[:, -4 + i, :]).unsqueeze(1)  # TODO don't hardcode 4
            nan_count = torch.isnan(embed_output).sum().item()
            if nan_count > 0:
                print(f"Time2VecEmbedding loop added {nan_count} NaNs at index {-4+i}")
            out.append(embed_output)
            # TODO have year be a continuous embedding since its not periodic

        # Concatenate results across the sequence length dimension
        x = torch.cat(out, dim=1)

        nan_count = torch.isnan(x).sum().item()
        print(f"Number of NaN values after embedding: {nan_count}")
        if nan_count > 0:
            print("got", nan_count, "nans")
            exit()

        x += self.categorical_embeddings

        nan_count = torch.isnan(x).sum().item()
        print(f"Number of NaN values after adding cat embedding: {nan_count}")
        if nan_count > 0:
            print("got", nan_count, "nans")
            exit()
        
        # TODO properly add batches across GPUs
        
        # TODO only add this if it seems like it improves performance, otherwise just unnecessary
        # x = self.decoder_input_proj(x)
        # nan_count = torch.isnan(x).sum().item()
        # print(f"Number of NaN values after decoder input proj: {nan_count}")
        x = self.append_cls(x)

        nan_count = torch.isnan(x).sum().item()
        print(f"Number of NaN values after appending cls: {nan_count}")
        if nan_count > 0:
            print("got", nan_count, "nans")
            exit()
        x_np = x.squeeze(0).detach().numpy()
        # Save numpy array to a text file
        np.savetxt("tensor_data.txt", x_np, fmt="%s", delimiter=",")
        
        x = self.decoder_input_norm
        nan_count = torch.isnan(x).sum().item()
        print(f"Number of NaN values after input layernorm and before block: {nan_count}")
        if nan_count > 0:
            print("got", nan_count, "nans")
            exit()
        
        depth = 0
        for blk in self.decoder_blocks:
            x = blk(x, attention_mask)
            # TODO this throws nans after 1-2 layers!!!
            # TODO change the initialization for the weights
            depth += 1
            print("cls token at depth", depth, x[:, 0, :])

            num_nans =  (torch.isnan(x).sum().item())
            if num_nans > 0:
                print("got", num_nans, "nans")
                exit()

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
        