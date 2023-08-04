from torch import nn
import torch
import utils
from classes import (
    TransformerBlock, ContinuousReverseEmbedding, Time2VecEmbedding, 
    TickerEmbedding, ContinuousEmbedding, CategoryEmbedding)
import numpy as np
import pandas as pd

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
        self.all_categories = self.categories_lookup["floats"] + self.categories_lookup["text"] + self.categories_lookup["datetime"]
        self.seq_len = len(self.all_categories) + 1 # +1 for the cls token
        
        self.df = None

        ## Get embeddings (original -> input embedding)
        self.embeddings_lookup = nn.ModuleDict()

        # These will not be in the df and have to be added during preprocessing  
        for category in self.categories_lookup["floats"]:
            num_days = utils.get_num_days(category)
            self.embeddings_lookup[category] = ContinuousEmbedding(num_days, self.embed_dim)
        
        # (Only ["Ticker"] for now)
        ticker_list = utils.get_ticker_list()
        self.ticker_embedder = TickerEmbedding(ticker_list, self.embed_dim)
        self.embeddings_lookup["Ticker"] = self.ticker_embedder
        
        # Encode the current datetime as a vector
        for category in self.categories_lookup["datetime"]:
            if category == "Year":
                time2vec_embedding = ContinuousEmbedding(1, self.embed_dim, bias=1900, scale=100) # not periodic
            elif category == "Month":
                time2vec_embedding = Time2VecEmbedding(12, self.embed_dim)
            elif category == "Day":
                time2vec_embedding = Time2VecEmbedding(31, self.embed_dim)
            elif category == "Weekday":
                time2vec_embedding = Time2VecEmbedding(7, self.embed_dim)
            elif category == "Hour":
                time2vec_embedding = Time2VecEmbedding(24, self.embed_dim)
            elif category == "Minute":
                time2vec_embedding = Time2VecEmbedding(60, self.embed_dim)
            else:
                raise ValueError(f"Time2vec Category {category} not recognized")
            
            self.embeddings_lookup[category] = time2vec_embedding
        
        # If we don't have values, don't do attention on it, but still need to add them all so the input seq is fixed
        self.attention_mask = torch.ones(self.seq_len).bool().expand(self.batch_size, -1).cuda()

        ## Get categorical embeddings and mask tokens to add to input embedding
        self.cat_embeddings = CategoryEmbedding(self.all_categories, embed_dim) # (same thing as pos emb for vit)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim)) * 0.02
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
        
        self.predictor = ContinuousReverseEmbedding(self.embed_dim, 1)
    
    def embed_original(self, df):
        self.df = df
        df_x = df.sample(self.batch_size)
        
        gt = 0. # TODO implement this
        
        all_data = {category: pd.Series(dtype='float64') for category in utils.get_base_categories()}
        for category in utils.get_base_categories():
            # Calculate returns wrt to previous value (100, 110, 99) -> (1, 1.1, 0.9)
            temp = df_x[category].pct_change() + 1
            temp.fillna(1, inplace=True) # By default the first value becomes nan bc no reference
            all_data[category] = temp

        # With the data from above, index out the correct days and then get the embedding for that
        # If we don't have the values for that window (for example if we have up to 600 days, it won't work for Volume_Last_512_1024_days) then
        # add zeros there instead
        x = []
        self.attention_mask = []
        for category_name in self.all_categories:
            category = utils.get_base_category_name(category_name)
            assert category in utils.get_base_categories(), f"Got category: {category}, category_name: {category_name}"
            parts = category_name.split('_')
            start, end = int(parts[-3]), int(parts[-2])
            # TODO implement the logic to get the correct data for each category
            if end > len(all_data[category]): # TODO how do we correspond "end" to the correct days? since num_rows != num_days
                embedded_data = torch.empty(self.batch_size, self.embed_dim).cuda()
                self.attention_mask.append(False)
            else:
                # TODO for indexing the days, just manually index each day to make sure it's within range, so there 
                # isn't an insertion/deletion mutation of the data that messes up everything
                # the code in the original get_window_data is good for this. use the builtin timedelta etc.
                
                # TODO figure out how to index the attention mask properly also, since we need it to be a 2d tensor along 
                # batch and N
                embedded_data = self.embeddings_lookup[category_name](all_data[category])
                embedded_data += self.cat_embeddings(category_name)
                self.attention_mask.append(True)
            x.append(embedded_data)
        
        x = torch.stack(x, dim=-1)
        x = x.permute(0, 2, 1)  # [batch_size, embed_dim, features] -> [batch_size, features, embed_dim]
        return x, gt
            
    def append_cls(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        self.attention_mask = [True] + self.attention_mask
        return x
    
    def forward_decoder(self, x):
        x = self.append_cls(x)
        x = self.decoder_input_proj(x)
        for blk in self.decoder_blocks:
            decoded_x = blk(decoded_x)
        decoded_x = self.decoder_norm(decoded_x)
        cls_token = decoded_x[:, 0, :]
        return cls_token
    
    def forward_loss(self, cls_token, gt):
        pred = self.predictor(cls_token)
        loss = utils.mean_squared_error(gt, pred)
        return loss
    
    def forward(self, df):
        x, gt = self.embed_original(df)
        cls_token = self.forward_decoder(x)
        loss = self.forward_loss(cls_token, gt)
        return loss
        