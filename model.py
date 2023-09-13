from torch import nn
import torch
import utils
from classes import (
    TransformerBlock, ContinuousUnembedding, Time2VecEmbedding, 
    TickerEmbedding, ContinuousEmbedding, CategoryEmbedding)
import numpy as np
import pandas as pd
import torch.nn.functional as F

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
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim)).cuda() * 0.02
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

    def get_values_for_range(self, df, category, current_date, start_days_back, end_days_back, ticker):
        
        # Convert current_date to a pandas Timestamp
        current_date = pd.Timestamp(current_date)

        # Compute the start and end dates for the range
        start_date = current_date - pd.offsets.BDay(start_days_back+1) # One day before the start
        end_date = current_date - pd.offsets.BDay(end_days_back+1)
        
        # Filter the DataFrame for the specified ticker
        subdf = df[df['Ticker'] == ticker]

        # Filter the DataFrame for the specified date range
        subdf = subdf[(subdf["Date"] >= start_date) & (subdf["Date"] <= end_date)]

        # Check if any date is missing within the range
        expected_length = start_days_back - end_days_back + 1
        if len(subdf) != expected_length:
            return None

        # Extract the values for the specified category
        values = subdf[category].values

        # Compute the returns (if any division by zero, set to zero)
        differences = values[1:] - values[:-1]
        returns = np.divide(differences, values[:-1], out=np.zeros_like(differences), where=values[:-1] != 0)

        # Check the length of the returns array
        if len(returns) != (start_days_back - end_days_back):
            return None

        # Cast to a tensor and return
        returns = torch.tensor(returns).float().cuda()
        return returns

    
    # TODO break into helpers
    def embed_original(self, df):

        # - sample the df for batch size rows
        df_x = df.sample(self.batch_size)
        gt = df_x["gt"].values # Percent change in TypicalPrice wrt previous day
        gt = torch.tensor(gt).cuda()
        
        # These will hold our results
        x = []
        attention_mask = []

        # Iterate over rows in df_x
        for index, row in df_x.iterrows(): # B dimension
            # These will hold the data for the current row
            row_data = []
            row_mask = []
            for category_window in utils.get_floats_categories(): # N dimension
                # Parse the category to get attribute and range of days
                category, (end_days_back, start_days_back) = utils.parse_category(category_window)

                # Extract values for the given attribute and range of days, considering only weekdays
                ticker = row["Ticker"]
                current_date = row["Date"]
                values = self.get_values_for_range(df, category, current_date, start_days_back, end_days_back, ticker)

                if values is None: # Missing dates, set attention mask to false
                    category_embedding = torch.zeros(self.embed_dim).cuda()
                    row_mask.append(False)
                else: # All dates present, set attention mask to true
                    category_embedding = self.embeddings_lookup[category_window](values)
                    category_embedding += self.cat_embeddings[category_window]

                    row_mask.append(True)
                
                if not any(row_mask):
                    print("WARNING", f"Row mask is all false for row with {current_date} and ticker {ticker}")
                    
                row_data.append(category_embedding)
            
            for category_window in utils.get_text_categories():
                ticker = row["Ticker"]
                category_embedding = self.embeddings_lookup["Ticker"](ticker)
                category_embedding += self.cat_embeddings[category_window]
                
                row_data.append(category_embedding)
                row_mask.append(True)
            
            for category_window in utils.get_time2vec_categories():
                time = row[category_window]
                time = torch.tensor([time]).float().cuda()
                category_embedding = self.embeddings_lookup[category_window](time)
                category_embedding += self.cat_embeddings[category_window]
                
                row_data.append(category_embedding)
                row_mask.append(True)
            
            # Convert row_data and row_mask to tensors
            row_data_tensor = torch.stack(row_data).cuda() # Stacking the tensor list
            row_mask_tensor = torch.tensor(row_mask, dtype=torch.bool).unsqueeze(-1).cuda() # Adding an extra dimension for [N, 1] shape

            # Append to the result holders
            x.append(row_data_tensor)
            attention_mask.append(row_mask_tensor) 
            
            print("finished index ", index)  
            
        # Convert lists of tensors to final tensors
        x = torch.stack(x).cuda()
        x = self.append_cls(x)
        attention_mask = torch.stack(attention_mask).squeeze(-1) # Removing the extra dimension after stacking

        # Create a tensor of True values with the same batch size
        true_tensor = torch.ones((self.batch_size, 1), dtype=torch.bool).cuda()

        # Concatenate the true_tensor with the attention_mask along dimension 1
        attention_mask = torch.cat([true_tensor, attention_mask], dim=1)
        
        assert x.shape == (self.batch_size, self.seq_len, self.embed_dim), f"Expected shape {(self.batch_size, self.seq_len, self.embed_dim)}, got {x.shape}"
        assert attention_mask.shape == (self.batch_size, self.seq_len), f"Expected shape {(self.batch_size, self.seq_len)}, got {attention_mask.shape}"
        assert gt.shape == (self.batch_size,), f"Expected shape {(self.batch_size,)}, got {gt.shape}"

        return x, attention_mask, gt
    
    def pad_x_for_embed_dim(self, tensor):
        padding_size = self.embed_dim - tensor.size(1)
        if padding_size > 0:
            tensor = F.pad(tensor, (0, padding_size))
        return tensor
    
    def forward_decoder(self, x, attention_mask=None):
        
        x = self.pad_x_for_embed_dim(x)
        print(x.shape, self.decoder_input_proj)
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
    
    def forward(self, x, attention_mask, gt):
        cls_token = self.forward_decoder(x, attention_mask)
        loss = self.forward_loss(cls_token, gt)
        return loss
        