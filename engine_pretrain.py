
import time
import numpy as np

def train_one_step(model, dataset, accum_iter, optimizer, batch_size_per_gpu):
    total_loss_for_step = 0.

    # zero the gradients at the start as we will be accumulating them over the entire step
    optimizer.zero_grad()
    
    losses_list = []

    for i in range(accum_iter):
        mini_batch_loss = 0.  # initialize the loss for this mini-batch

        for j in range(batch_size_per_gpu): # loop for batching
            a = time.time()
            x, mask, y = dataset[-1] # index is irrelevant
            b = time.time()
            # print("Took", b-a, "seconds to get data")
            loss = model(x, mask, y)
            c = time.time()
            # print("Took", c-b, "seconds to run fwd pass")
            mini_batch_loss += loss  # accumulate the loss tensors directly
            total_loss_for_step += loss.item()  # for logging purposes
            
            losses_list.append(loss.item())
            
            # print("On sample", i*batch_size_per_gpu + j)

        # call backward for the entire mini-batch
        mini_batch_loss = mini_batch_loss / (batch_size_per_gpu * accum_iter)
        d = time.time()
        mini_batch_loss.backward()
        e = time.time()
        print("Backward pass took", e-d, "seconds")

    # update the parameters after all gradients have been accumulated for the entire step
    optimizer.step()
    
    total_loss_for_step /= (batch_size_per_gpu * accum_iter)
    print("Total loss for the step:", total_loss_for_step)
    std_dev = np.std(losses_list)
    with open("losses.txt", "a") as f:
        f.write(f"Loss: {total_loss_for_step}, Std Dev: {std_dev}\n")

    return model, total_loss_for_step
