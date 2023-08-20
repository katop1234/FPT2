import torch
from torch import inf

def train_one_epoch(model, df, accum_iter, optimizer, batch_size_per_gpu):
    epoch_loss = 0.

    # zero the gradients
    optimizer.zero_grad()
    
    print("Starting one epoch")
    for i in range(accum_iter):
        x, loss, latents = model(df)
        epoch_loss += loss

    # calculate the backward pass
    epoch_loss.backward()
    print("Called backprop after one epoch. Got epoch loss of ", epoch_loss)

    # update the parameters
    optimizer.step()

    return model, epoch_loss


