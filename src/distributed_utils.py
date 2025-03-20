import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed(rank, world_size):
    """
    Setup distributed training environment
    
    Args:
        rank (int): The rank of the current process
        world_size (int): Total number of processes
    """
    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set device for the current process
    torch.cuda.set_device(rank)
    
    print(f"Initialized process {rank+1}/{world_size} on GPU {rank}")

def cleanup_distributed():
    """
    Clean up the distributed environment
    """
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process(rank):
    """
    Check if this is the main process (rank 0)
    
    Args:
        rank (int): Current process rank
        
    Returns:
        bool: True if this is the main process
    """
    return rank == 0

def get_model_parallel_rank():
    """
    Get the rank of the current process in the model parallel group
    
    Returns:
        int: Rank of current process
    """
    if dist.is_initialized():
        return dist.get_rank()
    return 0

def get_model_parallel_world_size():
    """
    Get the world size of the model parallel group
    
    Returns:
        int: World size
    """
    if dist.is_initialized():
        return dist.get_world_size()
    return 1
    
def wrap_model_distributed(model, rank):
    """
    Wrap model in DistributedDataParallel for multi-GPU training
    
    Args:
        model (torch.nn.Module): Model to wrap
        rank (int): Current process rank
        
    Returns:
        DDP: Wrapped model
    """
    # Ensure model is on the correct device
    model = model.to(rank)
    
    # Wrap model with DDP
    ddp_model = DDP(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False  # Set to True if needed
    )
    
    return ddp_model
