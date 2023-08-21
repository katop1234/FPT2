import torch
from torch import inf
import math, sys

def train_one_epoch(model, dataloader, accum_iter, optimizer, device):
    optimizer.zero_grad()
    total_loss = 0.0 # To accumulate loss

    for data_iter_step, (batch_data, batch_masks, batch_targets) in enumerate(dataloader):

        # Compute loss
        loss = model(x=batch_data, attention_mask=batch_masks, targets=batch_targets)
        loss_value = loss.item()

        # If the loss is not finite, skip the backward pass for this iteration
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, skipping this iteration")
            continue

        # Accumulate the gradients
        loss /= accum_iter
        loss.backward()
        total_loss += loss_value

        # If we've accumulated enough iterations, update the parameters
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad()

    print(f"Finished one epoch with total loss: {total_loss}")
