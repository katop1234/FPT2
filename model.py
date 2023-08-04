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
        self.categories_lookup["floats"] = utils.get_floats_categories()
        self.categories_lookup["text"] = utils.get_text_categories()
        self.categories_lookup["time2vec"] = utils.get_time2vec_categories()
        self.all_categories = self.categories_lookup["floats"] + self.categories_lookup["text"] + self.categories_lookup["time2vec"]
        
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
        
        ## Attention masking
        # If we don't have values, don't do attention on it, but still need to add it so the input seq is fixed
        self.missing_token = nn.Parameter(torch.zeros(self.embed_dim)) 
        self.attention_mask = [True * len(self.all_categories)]

        ## Get categorical embeddings and mask tokens to add to input embedding
        self.cat_embeddings = CategoryEmbedding(self.all_categories, embed_dim) # (same thing as pos emb for vit)
        
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
        
        ## Reverse embed decoded latent into original value
        self.predictor_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.reverse_embeddings = nn.ModuleDict()

        for category, feature_list in self.categories_lookup.items():
            if category == 'floats':
                for feature in feature_list:
                    num_days = utils.get_num_days(feature)
                    self.reverse_embeddings[feature] = ContinuousReverseEmbedding(self.embed_dim, num_days)
            elif category == 'text':
                for feature in feature_list:  # it should only be 'Ticker'
                    continue # TODO add once we know how to do this
                    self.reverse_embeddings[feature] = ContinuousReverseEmbedding(self.embed_dim, self.ticker_embedder.embed_dim)
        
                # self.ticker_loss = NegativeSamplingLoss(self.ticker_embedder)
                
        # floats category lookup to get the gt for loss calculation
        self.gtvalue_lookup = dict()
        
    def get_window_data(self, df_x, category_name):
        raise NotImplementedError
        '''
        Category name is in the format of "TypicalPrice_Last_0_10_days or Volume_Last_20_40_days"
        We need to find the df corresponding to the ticker in each row, and then append that list of values for each row
        '''
        # parse the window range from the category_name
        window_start, window_end = map(int, category_name.split('_')[-3:-1])

        # initialize empty list to collect data tensors
        tensors_list = []

        # convert df_x date to datetime format
        df_x['Date'] = pd.to_datetime(df_x['Date'])

        for idx, row in df_x.iterrows():
            ticker, current_date = row['Ticker'], row['Date']

            # load the data for this ticker
            df_ticker = utils.read(f"ticker_data/{ticker}_data.ser")
            
            df_ticker['Date'] = pd.to_datetime(df_ticker['Date'])
            df_ticker.set_index('Date', inplace=True)

            # calculate the start and end dates
            start_date = current_date - pd.Timedelta(days=window_end)
            end_date = current_date - pd.Timedelta(days=window_start) - pd.Timedelta(days=1)  # subtract 1 day to exclude end_date

            # create the windowed data
            window_data = df_ticker.loc[start_date:end_date]  # slice with dates
            window_data = window_data[category_name.split('_')[0]]

            # Reindex to a complete date range and fill missing values
            all_dates = pd.date_range(start=start_date, end=end_date)
            window_data = window_data.reindex(all_dates, fill_value=utils.MISSING_VALUE)

            # convert to torch tensor and append to list
            tensor_data = torch.tensor(window_data.values, dtype=torch.float32)
            tensors_list.append(tensor_data)

        # concatenate all tensors along the batch dimension
        category_data = torch.stack(tensors_list, dim=0).cuda()

        return category_data
    
    def embed_original(self, df):
        '''
        Preprocess, mask 15% of inputs and get data tensor of embeddings x from the df
        1 is masked, 0 if not masked
        '''
        self.df = df
        df_x = df.sample(self.batch_size)
        
        x = []
        all_categories = self.all_categories
        
        for category_name in all_categories:
            if category_name in self.categories_lookup["floats"]:
                # category_data = self.get_window_data(df_x, category_name)
                
                # get all data for this category
                # 
                pass
            elif category_name in self.categories_lookup["text"]:
                category_data = df_x[category_name]
            else:
                raise ValueError(f"Category {category_name} not recognized")

            embedded_data = self.embeddings_lookup[category_name](category_data)
            embedded_data += self.cat_embeddings[category_name]
            x.append(embedded_data)

        x = torch.stack(x, dim=-1)
        x = x.permute(0, 2, 1)  # [batch_size, embed_dim, features] -> [batch_size, features, embed_dim]

        return x
    
    def forward_loss(self, pred_lookup):
        loss = 0.
        
        self.gt_value_lookup = dict()
        self.df = None
        
        return loss
    
    def forward(self, df_x):
        x = self.embed_original(df_x)
        decoded_x = self.forward_decoder(x)
        pred_lookup = self.reverse_embed_to_original(decoded_x)
        loss = self.forward_loss(pred_lookup)
        # true_predictions = self.get_predictions(x_pred_dict) # Normalizing undone
        return x, loss
        