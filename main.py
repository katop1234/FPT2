import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import utils
import os
from model import FPT
from engine_pretrain import train_one_epoch
from dataset_factory import FinancialDataset

# Hyperparameters
num_epochs = 1000
total_batch_size = 1024
batch_size_per_gpu = 64
lr = 1.6e-3
embed_dim = 256
depth = 16

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

    model = FPT(embed_dim=embed_dim,
                 depth=depth,
                 batch_size=batch_size_per_gpu,
                )
    
    if num_gpus > 0:
        model = model.cuda(gpu)

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

    for epoch in range(num_epochs):
        train_one_epoch(model, dataset, accum_iter, optimizer, batch_size_per_gpu)

def main():
    ngpus_per_node = torch.cuda.device_count()
    print(f"Using {ngpus_per_node} GPUs")
    if ngpus_per_node == 0:
        main_worker(0, 0)
    else:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node,))

if __name__ == "__main__":
    main()
