
from torch import inf
import constants

def train_one_epoch(model, dataset, accum_iter, optimizer, batch_size_per_gpu):
    epoch_loss = 0.

    # zero the gradients
    optimizer.zero_grad()
    
    print("Starting one epoch")
    losses = []
    for i in range(accum_iter):
        for _ in range(batch_size_per_gpu): # TODO implement batching lol
            x, mask, y = dataset[i]
            loss = model(x, mask, y)
            epoch_loss += loss

            losses.append(loss.item())

    # calculate the backward pass
    print('calling backprop!')
    epoch_loss.backward()
    print("Called backprop after one epoch. Got epoch loss of ", epoch_loss)
    print("With losses", losses)

    # update the parameters
    optimizer.step()

    return model, epoch_loss
