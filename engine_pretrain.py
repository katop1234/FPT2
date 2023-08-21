import torch
from torch import inf

def train_one_epoch(model, dataloader, accum_iter, optimizer):
    epoch_loss = 0.

    # zero the gradients
    optimizer.zero_grad()
    
    print("Starting one epoch")
    
    # Iterate through the batches from the DataLoader
    for batch_idx, (batch_data, batch_masks) in enumerate(dataloader):
        
        # If using GPU, move the batch to the GPU
        if torch.cuda.is_available():
            batch_data = batch_data.cuda()
            batch_masks = batch_masks.cuda()

        for i in range(accum_iter):
            # Pass the batch through the model
            x, loss, latents = model(batch_data, attention_mask=batch_masks)
            epoch_loss += loss.item() / accum_iter

        # Divide the loss by accum_iter and compute gradients
        (loss / accum_iter).backward()
        
        # Update only every accum_iter
        if (batch_idx + 1) % accum_iter == 0:
            print("Called backprop for batch. Accumulated loss: ", epoch_loss)
            optimizer.step()
            optimizer.zero_grad()

    print("Finished epoch with total loss of ", epoch_loss)

    return model, epoch_loss



