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
    
    def append_cls(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        return x
    
    # TODO break into helpers
    def embed_original(self, df):
        # - sample the df for batch size rows
        df_x = df.sample(self.batch_size)
        gt = df_x["gt"].values # Percent change in TypicalPrice wrt previous day
        
        # These will hold our results
        x = []
        attention_mask = []

        # Iterate over rows in df_x
        for index, row in df_x.iterrows(): # B dimension
            # These will hold the data for the current row
            row_data = []
            row_mask = []

            # Iterate over categories
            for category in utils.get_floats_categories(): # N dimension
                # Split category name to get base category and timeframe
                base_category, (start, end) = utils.parse_category(category)

                # Get the current date from the row
                current_date = row['Date']

                # Calculate start_date as start + 1 business days before the current date
                start_date = current_date - pd.offsets.BDay(start + 1)

                # Calculate end_date as end business days before the current date
                end_date = current_date - pd.offsets.BDay(end)

                # Generate all business dates between start_date and end_date
                date_range = pd.date_range(start_date, end_date, freq='B')

                # Check if all dates in the range exist in the DataFrame for the current ticker
                ticker = row['Ticker']
                if all(date in df['Date'].values and df['Ticker'].values == ticker for date in date_range):
                    # Filter the DataFrame for the specific ticker and date range
                    filtered_data = df[(df['Date'] >= start_date) & (df['Date'] <= end_date) & (df['Ticker'] == ticker)]

                    # Get the values for the base_category column
                    data = filtered_data[base_category].values

                    # Calculate percentage changes
                    returns = (data[1:] - data[:-1]) / data[:-1]
                    returns = torch.tensor(returns).cuda()

                    # Get category embedding
                    category_embedding = self.embeddings_lookup[category](returns)
                    category_embedding += + self.cat_embeddings[category]

                    # Add True to row_mask
                    row_mask.append(True)
                else:
                    # Create a zero tensor for category embedding
                    category_embedding = torch.zeros(self.embed_dim)

                    # Add False to row_mask
                    row_mask.append(False)

                # Add to row data
                row_data.append(category_embedding)

            # Convert row_data to a 2D tensor and add to category data
            x.append(torch.stack(row_data))

            # Convert row_mask to a 1D tensor and add to attention_mask
            attention_mask.append(torch.tensor(row_mask))

        # Convert category_data to a 3D tensor
        x = torch.stack(x)
        x = self.append_cls(x)

        # Add True to each row in attention_mask to account for the [CLS] token
        attention_mask = [torch.cat([torch.tensor(True), row]) for row in attention_mask]

        # Convert attention_mask to a 2D tensor
        attention_mask = torch.stack(attention_mask)
        
        assert x.shape == (self.batch_size, self.seq_len, self.embed_dim)
        assert attention_mask.shape == (self.batch_size, self.seq_len)
        assert gt.shape == (self.batch_size,)

        return x, attention_mask, gt
    
    def forward_decoder(self, x, attention_mask=None):
        x = self.decoder_input_proj(x)
        for blk in self.decoder_blocks:
            x = blk(x, attention_mask)
        x = self.decoder_norm(x)
        cls_token = x[:, 0, :]
        return cls_token
    
    def forward_loss(self, cls_token, gt):
        pred = self.predictor(cls_token)
        loss = utils.mean_squared_error(gt, pred)
        return loss
    
    def forward(self, df):
        x, attention_mask, gt = self.embed_original(df)
        cls_token = self.forward_decoder(x, attention_mask)
        loss = self.forward_loss(cls_token, gt)
        return loss
        