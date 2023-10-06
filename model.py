from torch import nn
import torch
import utils
from classes import TransformerBlock, Time2VecEmbedding
import numpy as np
import pandas as pd
import torch.nn.functional as F
import constants

device = constants.device
utils.fix_random_seed()

class FPT(nn.Module):
    def __init__(self,
                 embed_dim=256,
                 depth=32,
                 input_dim=256,
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.input_dim = input_dim
        
        self.categories_lookup = dict()
        self.categories_lookup["floats"] = utils.get_floats_categories()
        self.categories_lookup["text"] = utils.get_text_categories()
        self.categories_lookup["datetime"] = utils.get_time2vec_categories()
        self.all_categories = self.categories_lookup["floats"] + self.categories_lookup["datetime"] # + self.categories_lookup["text"] 
        self.seq_len = len(self.all_categories) + 1 # +1 for the cls token
        
        # (Only ["Ticker"] for now)
        self.ticker_list = utils.get_ticker_list()

        ## Embed raw tokens
        self.sined_time_feats = Time2VecEmbedding(embed_dim)
        
        self.input_linear_projection = nn.Linear(input_dim, embed_dim)
        
        ## Get categorical embeddings and mask tokens to add to input embedding
        # TODO try having a shared "base" initialized vector for each categorical embedding
        # which we repeat over self.seq_len - 1 times 
        # then each category's embedding = base_cat_embedding_vector + cat_specific_embedding
        # base_embedding init is randn(0.02), let cat_specific_embedding init be randn(0.001)
        # This is so categories that don't get seen that much start off around this base vector and the
        # random noise doesn't overwrite the input's information that much
        self.base_categorical_embedding = nn.Parameter(torch.randn(1, self.embed_dim).repeat(self.seq_len - 1, 1) * 0.02).to(device)
        
        self.categorical_embeddings = nn.Parameter(torch.randn(self.seq_len - 1, self.embed_dim) * 0.001).to(device) # remove 1 for cls
        self.categorical_embeddings += self.base_categorical_embedding
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02).to(device)
        
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                )
                for _ in range(depth)
            ]
        )
        
        self.predictor = nn.Linear(self.embed_dim, 1)

        # TODO i'm interested in seeing if a identity + randn() init works better for linear projections
        # try ablation on that too over a 0-mean one
        def initialize_weights(module):
            if isinstance(module, (nn.Linear)):
                module.weight.data.normal_(mean=0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

        self.apply(initialize_weights)
        
        # For debugging
        self.counter = 0
        self.print_freq = 256
    
    def append_cls(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        return x
    
    def forward_decoder(self, x, attention_mask=None):
        
        x_np = x.cpu().squeeze(0).detach().numpy()
        np.savetxt("input_x.txt", x_np, fmt="%s", delimiter=",")

        nan_count = torch.isnan(x).sum().item()
        
        if nan_count > 0:
            print(f"Number of NaN values in input: {nan_count}")
            exit()
        
        # TODO replace this with continuous embedding such that it 
        # just is Linear(input_dim, embed_dim)
        x = x.unsqueeze(0)

        # Split the tensor based on your token types
        # TODO change hardcoding on this!
        non_time_tokens = x[:, :-4, :]
        time_tokens = x[:, -4:, :]
        sined_time_tokens = self.sined_time_feats(time_tokens)
        x = torch.cat([non_time_tokens, sined_time_tokens], dim=1)
        
        # TODO try without these and see if it even matters!
        # TODO i'm concerned about padding the shorter sequences with all 0s...
        # feels like they might introduce a bunch of useless information?
        # see if this is an issue by probing into trained model's parameters and 
        # what it's doing, otherwise don't overengineer. feels off but i think
        # transformers are smart enough to deal with it on their own.
        x = self.input_linear_projection(x)

        x += self.categorical_embeddings
        
        x_np = x.cpu().squeeze(0).detach().numpy()
        np.savetxt("x_after_add_cat_emb.txt", x_np, fmt="%s", delimiter=",")

        nan_count = torch.isnan(x).sum().item()
        
        if nan_count > 0:
            print(f"Number of NaN values after adding cat embedding: {nan_count}")
            exit()
        
        # TODO properly add batches across GPUs
        
        x = self.append_cls(x)

        nan_count = torch.isnan(x).sum().item()
        
        if nan_count > 0:
            print(f"Number of NaN values after appending cls: {nan_count}")
            exit()
        
        np.savetxt("transformer_input_tensor_data.txt", x_np, fmt="%s", delimiter=",")
        
        depth = 0
        for blk in self.decoder_blocks:
            if self.counter % self.print_freq == 0: 
                input_tensor = x.clone()  # Store the input tensor before processing
                
            x = blk(x, attention_mask)
            
            # Compute cosine similarity
            if self.counter % self.print_freq == 0:
                cos_sim = F.cosine_similarity(input_tensor.flatten(), x.flatten(), dim=0)
                print(f"Cosine Similarity at depth {depth}: {cos_sim.item()}", "Samples seen so far:", self.counter)
            
            depth += 1

            num_nans =  (torch.isnan(x).sum().item())
            if num_nans > 0:
                print("got", num_nans, "nans after transforemr block", depth)
                exit()

        cls_token = x[:, 0, :]
        return cls_token
    
    def forward_loss(self, cls_token, gt):
        pred = self.predictor(cls_token)
        loss = utils.mean_squared_error(gt, pred)
        return loss
    
    def forward(self, x, attention_mask, gt):
        # TODO also change the Time2Vec embedding class to never worry abt modulus other than 1

        # TODO for datetime vectors, just copy the value over repeatedly, don't give it a bunch of 0s

        # TODO consider making a single token for datetime values

        cls_token = self.forward_decoder(x, attention_mask)
        loss = self.forward_loss(cls_token, gt)
        # print("Finished a forward pass and got a loss of", loss.item())
        
        self.counter += 1
        
        return loss
        