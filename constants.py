import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hardcoded to give ~N(0,1) distribution of gt for better model prediction
gt_mean, gt_std = (0.0006015757566498649, 0.023403779903275895)

num_tokens_back = 5
