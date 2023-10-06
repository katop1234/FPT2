import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import constants
import os
from model import FPT
from engine_pretrain import train_one_step
from dataset_factory import FinancialDataset

# Hyperparameters
num_steps = 1000
total_batch_size = 256
batch_size_per_gpu = 128
lr = 1e-5
input_dim = 256
embed_dim = 256
depth = 8
checkpt_freq = 25

# Variables
num_gpus = torch.cuda.device_count()
if num_gpus == 0:
    accum_iter = total_batch_size // (batch_size_per_gpu)
else:
    accum_iter = total_batch_size // (batch_size_per_gpu * num_gpus)

def main_worker(gpu, ngpus_per_node):
    os.environ['RANK'] = str(gpu)
    os.environ['WORLD_SIZE'] = str(ngpus_per_node)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    if num_gpus > 1:
        torch.cuda.set_device(gpu)
        torch.distributed.init_process_group(backend='nccl')

    # TODO add weight decay and dropout later
    model = FPT(embed_dim=embed_dim,
                depth=depth,
                input_dim=input_dim,
                )
    
    # Ensure the directory exists
    config_folder = f"batch_{total_batch_size}_lr_{lr}_input_{input_dim}_embed_{embed_dim}_depth_{depth}"
    checkpoint_dir = os.path.join("./serialized/checkpoints/", config_folder)

    # Ensure the directory exists
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"The model has {total_params:,} parameters.")
    
    # Calculate the total number of trainable parameters
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {total_trainable_params:,} trainable parameters.")

    if num_gpus > 0:
        model = model.cuda()
        print("Moved model to the gpu!")

    if num_gpus > 1:
        model = DDP(model, device_ids=[gpu])
        model._set_static_graph()
    
    # check if main prcoess then only print
    print(f"Using GPU: {gpu} ")
    if gpu == 0: print("using accum iter of ", accum_iter)
    
    eff_batch_size = batch_size_per_gpu * ngpus_per_node * accum_iter
    eff_learning_rate = lr * eff_batch_size / 1024
    optimizer = torch.optim.Adam(model.parameters(), lr=eff_learning_rate)

    dataset = FinancialDataset("SNPdata.ser")

    print("Using batch size per gpu", batch_size_per_gpu, "accum iter", accum_iter)
    # TODO make sure this uses distributed training!
    for step in range(num_steps):
        train_one_step(model, dataset, accum_iter, optimizer, batch_size_per_gpu)
        
        if step % checkpt_freq == 0:
            # Construct the filename with just the step count
            filename = f"model_step_{step}.pt"
            filepath = os.path.join(checkpoint_dir, filename)
            
            # Save the model
            torch.save(model.state_dict(), filepath)

def main():
    ngpus_per_node = torch.cuda.device_count()
    print(f"Using {ngpus_per_node} GPUs")
    if ngpus_per_node == 0:
        main_worker(0, 0)
    else:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node,))

if __name__ == "__main__":
    main()
