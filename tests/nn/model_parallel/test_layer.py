import torch.multiprocessing as mp
import torch.distributed as dist
import os

# FIXME(tom) copied from test_oss.py, should factor into common code
def dist_init(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def run_test_layer(rank, world_size):
    dist_init(rank, world_size)
    print(f"run_test_layer {rank}, {world_size}")
    assert rank in [0,1,2]

def test_add_param_group():
    world_size = 3
    mp.spawn(run_test_layer, args=(world_size,), nprocs=world_size, join=True)
