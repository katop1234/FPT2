from torch import nn
import torch
import utils
from classes import TransformerBlock, ContinuousReverseEmbedding, NegativeSamplingLoss, TickerEmbedding, ContinuousEmbedding, CategoryEmbedding
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
        self.categories_lookup["floats"] = utils.get_window_categories()
        self.categories_lookup["text"] = utils.get_text_categories()
        self.category_list = self.categories_lookup["floats"] + self.categories_lookup["text"]

        ## Get embeddings (original -> input embedding)
        self.embeddings = nn.ModuleDict()

        # These will not be in the df and have to be added during preprocessing  
        for category in self.categories_lookup["floats"]:
            num_days = utils.get_num_days(category)
            self.embeddings[category] = ContinuousEmbedding(num_days, self.embed_dim)
        
        # (Only ["Ticker"] for now)
        ticker_list = utils.get_ticker_list()
        self.ticker_embedder = TickerEmbedding(ticker_list, self.embed_dim)
        self.embeddings["Ticker"] = self.ticker_embedder

        ## Get categorical embeddings and mask tokens to add to input embedding
        self.cat_embeddings = CategoryEmbedding(self.category_list, embed_dim) # (same thing as pos emb for vit)
        
        self.decoder_input_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.decoder_norm = nn.LayerNorm(self.embed_dim)
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                )
                for _ in range(decoder_depth)
            ]
        )
        
        ## Reverse embed decoded latent into original value
        self.predictor_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.reverse_embeddings = nn.ModuleDict()

        for category, feature_list in self.categories_lookup.items():
            if category == 'float':
                for feature in feature_list:
                    self.reverse_embeddings[feature] = ContinuousReverseEmbedding(self.embed_dim, 1)
            elif category == 'text':
                for feature in feature_list:  # it should only be 'Ticker'
                    continue # TODO add once we know how to do this
                    self.reverse_embeddings[feature] = ContinuousReverseEmbedding(self.embed_dim, self.ticker_embedder.embed_dim)
        
                # self.ticker_loss = NegativeSamplingLoss(self.ticker_embedder)
            elif category == 'window':
                for feature in feature_list:
                    num_days = utils.get_num_days(feature)
                    self.reverse_embeddings[feature] = ContinuousReverseEmbedding(self.embed_dim, num_days)
                
        # window category lookup to get the gt for loss calculation
        self.gtvalue_lookup = {}
        
        # Because some categories' values are too large, we need to scale them up later
        # We expect the neural network to predict values from 0 to 1
        self.scale_factors = {
            "Open": torch.tensor(1e4),
            "High": torch.tensor(1e4),
            "Low": torch.tensor(1e4),
            "Close": torch.tensor(1e4),
            "Adj Close": torch.tensor(1e4),
            "TypicalPrice": torch.tensor(1e4),
            "Volume": torch.tensor(1e7),
            "Year": torch.tensor(2050.0),
            "Month": torch.tensor(12.0),
            "Day": torch.tensor(31.0),
        }
        
        for category in self.scale_factors:
            self.scale_factors[category].requires_grad = False # TODO do we want this to be learnable
            # if it's learnable, i'm concerned it could heavily affect the mag of the gradients getting backpropped
        
        for full_category_name in self.categories_lookup["window"]:
            category = utils.get_category_name(full_category_name) # Remove the days part
            self.scale_factors[full_category_name] = self.scale_factors[category]
    
    def forward(self, df_x):
        x, mask = self.embed_original(df_x)
        decoded_x = self.forward_decoder(x, mask)
        pred_lookup = self.reverse_embed_to_original(decoded_x)
        loss = self.forward_loss(pred_lookup)
        # true_predictions = self.get_predictions(x_pred_dict) # Normalizing undone
        return x, loss, latents
        