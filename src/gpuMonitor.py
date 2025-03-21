import torch

def get_gpu_memory_usage():
    """
    Get the current GPU memory usage.
    
    Returns:
        tuple: (allocated_memory_GB, cached_memory_GB, total_memory_GB)
    """
    if not torch.cuda.is_available():
        return (0, 0, 0)
    
    device = torch.cuda.current_device()
    
    # Get memory statistics
    allocated_bytes = torch.cuda.memory_allocated(device)
    cached_bytes = torch.cuda.memory_reserved(device)
    total_bytes = torch.cuda.get_device_properties(device).total_memory
    
    # Convert to GB
    allocated_gb = allocated_bytes / (1024 ** 3)
    cached_gb = cached_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)
    
    return (allocated_gb, cached_gb, total_gb)

def print_gpu_memory_usage(prefix=""):
    """
    Print the current GPU memory usage.
    
    Args:
        prefix (str): Optional prefix to add before the memory usage output
    """
    if not torch.cuda.is_available():
        print(f"{prefix}GPU not available")
        return
    
    allocated_gb, cached_gb, total_gb = get_gpu_memory_usage()
    
    print(f"{prefix}GPU Memory: {allocated_gb:.2f}GB allocated, "
          f"{cached_gb:.2f}GB cached, "
          f"{total_gb:.2f}GB total, "
          f"{(allocated_gb/total_gb)*100:.1f}% used")

def log_memory_snapshot(step_name=""):
    """
    Log a memory snapshot with a descriptive step name.
    
    Args:
        step_name (str): Name of the current step or operation
    """
    if not torch.cuda.is_available():
        return
    
    print(f"[MEMORY] {step_name} - ", end="")
    print_gpu_memory_usage()
    
    # Optional: force garbage collection
    # import gc
    # gc.collect()
    # torch.cuda.empty_cache()