
import time
import numpy as np
import torch

def train_one_step(model, dataset, accum_iter, optimizer, batch_size_per_gpu):
    total_loss_for_step = 0.

    # zero the gradients at the start as we will be accumulating them over the entire step
    optimizer.zero_grad()
    
    losses_list = []

    for i in range(accum_iter):
        mini_batch_loss = 0.  # initialize the loss for this mini-batch

        X = []
        MASKS = []
        Y = []
        a = time.time()
        for j in range(batch_size_per_gpu): # loop for batching
            x, mask, y = dataset[-1] # index is irrelevant
            X.append(x)
            MASKS.append(mask)
            Y.append(y)
        
        X = torch.stack(X, dim=0)        # Stacking along a new dimension (batch dimension)
        MASKS = torch.stack(MASKS, dim=0)
        Y = torch.stack(Y, dim=0)
        b = time.time()
        print("Took", b-a, "seconds to get all the data")
        loss = model(X, MASKS, Y)
        c = time.time()
        print("Took", c-b, "seconds to run fwd pass")
        
        loss /= accum_iter
        loss.backward()
        d = time.time()
        print("Took", d-c, "seconds to run backward pass")
        
        total_loss_for_step += loss.item()

    # update the parameters after all gradients have been accumulated for the entire step
    optimizer.step()
    
    print("Total loss for the step:", total_loss_for_step)
    losses_std = 0 # np.std(losses_list)
    
    return total_loss_for_step, losses_std
