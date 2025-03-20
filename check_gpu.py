#!/usr/bin/env python
"""
GPU Diagnostic Tool for Multi-GPU Training
This script checks the system's capabilities for multi-GPU training.
"""

import torch
import os
import sys
import platform
import psutil
from tabulate import tabulate

def get_cuda_info():
    """Get CUDA version and available GPU information."""
    cuda_info = {}
    try:
        if torch.cuda.is_available():
            cuda_info["CUDA Available"] = "Yes"
            cuda_info["CUDA Version"] = torch.version.cuda
            cuda_info["cuDNN Version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "Not available"
            cuda_info["Number of GPUs"] = torch.cuda.device_count()
            
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                cuda_info[f"GPU {i} - Name"] = device_props.name
                cuda_info[f"GPU {i} - VRAM"] = f"{device_props.total_memory / (1024**3):.2f} GB"
                cuda_info[f"GPU {i} - Compute Capability"] = f"{device_props.major}.{device_props.minor}"
                cuda_info[f"GPU {i} - Multi Processor Count"] = device_props.multi_processor_count
        else:
            cuda_info["CUDA Available"] = "No"
    except Exception as e:
        cuda_info["Error"] = str(e)
    
    return cuda_info

def get_system_info():
    """Get system information."""
    system_info = {}
    system_info["System"] = platform.system()
    system_info["Release"] = platform.release()
    system_info["Version"] = platform.version()
    system_info["Machine"] = platform.machine()
    system_info["Processor"] = platform.processor()
    system_info["CPU Count"] = psutil.cpu_count(logical=False)
    system_info["Logical CPU Count"] = psutil.cpu_count(logical=True)
    system_info["Total RAM"] = f"{psutil.virtual_memory().total / (1024**3):.2f} GB"
    system_info["Available RAM"] = f"{psutil.virtual_memory().available / (1024**3):.2f} GB"
    
    return system_info

def get_nccl_info():
    """Check NCCL availability for distributed training."""
    nccl_info = {}
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            # Check if NCCL is available
            if hasattr(torch.distributed, "is_nccl_available"):
                nccl_info["NCCL Available"] = "Yes" if torch.distributed.is_nccl_available() else "No"
            else:
                # Older PyTorch versions
                nccl_info["NCCL Available"] = "Unknown (older PyTorch version)"
            
            # Check if distributed is available
            nccl_info["Distributed Module Available"] = "Yes" if hasattr(torch, "distributed") else "No"
            
            # Check backend support
            if hasattr(torch.distributed, "is_mpi_available"):
                nccl_info["MPI Available"] = "Yes" if torch.distributed.is_mpi_available() else "No"
            
            if hasattr(torch.distributed, "is_gloo_available"):
                nccl_info["Gloo Available"] = "Yes" if torch.distributed.is_gloo_available() else "No"
        else:
            nccl_info["NCCL Available"] = "Not applicable (no multiple GPUs found)"
    except Exception as e:
        nccl_info["Error"] = str(e)
    
    return nccl_info

def check_distributed_launch():
    """Check if torch.distributed.launch is available."""
    try:
        import torch.distributed.launch
        return "Available"
    except ImportError:
        return "Not available"
    except Exception as e:
        return f"Error: {str(e)}"

def get_torch_info():
    """Get PyTorch version information."""
    torch_info = {}
    torch_info["PyTorch Version"] = torch.__version__
    torch_info["PyTorch built with CUDA"] = torch.cuda.is_available()
    torch_info["torch.distributed.launch"] = check_distributed_launch()
    
    try:
        # Try to import DeepSpeed
        import deepspeed
        torch_info["DeepSpeed Available"] = "Yes"
        torch_info["DeepSpeed Version"] = deepspeed.__version__
    except ImportError:
        torch_info["DeepSpeed Available"] = "No"
    except Exception as e:
        torch_info["DeepSpeed Error"] = str(e)
    
    return torch_info

def check_mpi():
    """Check for MPI support."""
    try:
        from mpi4py import MPI
        return {
            "MPI Available": "Yes",
            "MPI Version": MPI.Get_version()
        }
    except ImportError:
        return {"MPI Available": "No"}
    except Exception as e:
        return {"MPI Error": str(e)}

def get_environment_vars():
    """Get relevant environment variables for distributed training."""
    env_vars = {}
    
    important_vars = [
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "NCCL_DEBUG",
        "NCCL_SOCKET_IFNAME",
        "MASTER_ADDR",
        "MASTER_PORT",
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "NODE_RANK",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS"
    ]
    
    for var in important_vars:
        env_vars[var] = os.environ.get(var, "Not set")
    
    return env_vars

def run_gpu_test():
    """Run a simple GPU test to check if CUDA operations work."""
    if not torch.cuda.is_available():
        return {"GPU Test": "CUDA not available"}
    
    try:
        # Test a basic operation on each GPU
        results = {}
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            # Create tensors on this GPU
            x = torch.rand(1000, 1000, device=f"cuda:{i}")
            y = torch.rand(1000, 1000, device=f"cuda:{i}")
            # Perform matrix multiplication
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            z = torch.mm(x, y)
            end.record()
            
            # Sync and get time
            torch.cuda.synchronize()
            elapsed_time = start.elapsed_time(end)
            results[f"GPU {i} Test"] = "Passed"
            results[f"GPU {i} Operation Time"] = f"{elapsed_time:.3f} ms"
            
            # Test memory allocation
            try:
                # Try to allocate a large tensor to test memory
                large_tensor = torch.zeros(int(torch.cuda.get_device_properties(i).total_memory * 0.7 // 4), device=f"cuda:{i}")
                del large_tensor
                results[f"GPU {i} Memory Test"] = "Passed"
            except Exception as e:
                results[f"GPU {i} Memory Test"] = f"Failed: {str(e)}"
            
            # Clear cache
            torch.cuda.empty_cache()
            
        return results
    except Exception as e:
        return {"GPU Test": f"Failed: {str(e)}"}

def check_multi_gpu_communication():
    """Test basic communication between GPUs if multiple are available."""
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        return {"Multi-GPU Communication": "Not applicable (less than 2 GPUs)"}
    
    try:
        results = {}
        # Create tensors on different GPUs
        tensor1 = torch.rand(100, 100, device="cuda:0")
        tensor2 = torch.zeros(100, 100, device="cuda:1")
        
        # Copy data between GPUs
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        tensor2.copy_(tensor1)
        end.record()
        
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
        
        # Verify the copy worked
        same = torch.all(tensor1.cpu() == tensor2.cpu()).item()
        results["GPU to GPU Transfer"] = "Passed" if same else "Failed"
        results["GPU to GPU Transfer Time"] = f"{elapsed_time:.3f} ms"
        
        # Test distributed initialization if available
        try:
            torch.distributed.init_process_group(
                backend="nccl",
                init_method="tcp://localhost:12355",
                rank=0,
                world_size=1
            )
            torch.distributed.destroy_process_group()
            results["NCCL Initialization"] = "Successful"
        except Exception as e:
            results["NCCL Initialization"] = f"Failed: {str(e)}"
        
        return results
    except Exception as e:
        return {"Multi-GPU Communication": f"Failed: {str(e)}"}

def main():
    """Main function to collect and display system information."""
    print("GPU and Distributed Training Diagnostic Tool")
    print("=" * 50)
    
    info_sections = [
        ("System Information", get_system_info()),
        ("PyTorch Information", get_torch_info()),
        ("CUDA Information", get_cuda_info()),
        ("NCCL Information", get_nccl_info()),
        ("MPI Information", check_mpi()),
        ("Environment Variables", get_environment_vars())
    ]
    
    # Print all collected information
    for section_name, section_data in info_sections:
        print(f"\n{section_name}")
        print("-" * 50)
        if section_data:
            table_data = [[key, value] for key, value in section_data.items()]
            print(tabulate(table_data, tablefmt="plain"))
        else:
            print("No data available")
    
    # Run GPU tests if CUDA is available
    if torch.cuda.is_available():
        print("\nRunning GPU Tests")
        print("-" * 50)
        gpu_test_results = run_gpu_test()
        table_data = [[key, value] for key, value in gpu_test_results.items()]
        print(tabulate(table_data, tablefmt="plain"))
        
        if torch.cuda.device_count() > 1:
            print("\nTesting Multi-GPU Communication")
            print("-" * 50)
            multi_gpu_results = check_multi_gpu_communication()
            table_data = [[key, value] for key, value in multi_gpu_results.items()]
            print(tabulate(table_data, tablefmt="plain"))
    
    print("\nDiagnostic Summary")
    print("-" * 50)
    
    # Provide a summary of readiness for distributed training
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Multi-GPU training is not possible.")
    elif torch.cuda.device_count() < 2:
        print(f"⚠️ Only {torch.cuda.device_count()} GPU detected. Multi-GPU training requires at least 2 GPUs.")
    else:
        print(f"✅ {torch.cuda.device_count()} GPUs detected and ready for distributed training.")
    
    if hasattr(torch.distributed, "is_nccl_available") and torch.distributed.is_nccl_available():
        print("✅ NCCL is available for optimized GPU communication.")
    else:
        print("⚠️ NCCL may not be available. Check PyTorch installation.")
    
    # Provide recommendations
    print("\nRecommendations:")
    if torch.cuda.device_count() >= 2:
        print("1. Use 'python -m torch.distributed.launch --nproc_per_node={} src/main_distributed.py' to start training.".format(torch.cuda.device_count()))
        print("2. Ensure your batch size is appropriate for your GPU memory.")
        print("3. Consider using gradient accumulation for larger effective batch sizes.")
        print("4. Set environment variable NCCL_DEBUG=INFO for detailed NCCL logs if needed.")
    else:
        print("1. Use single GPU training with 'python src/main.py'")
        print("2. For distributed training, ensure you have multiple GPUs available.")
    
    # Memory recommendations
    if torch.cuda.is_available():
        total_memory = sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count()))
        total_memory_gb = total_memory / (1024**3)
        
        if total_memory_gb < 16:
            print("⚠️ Limited GPU memory detected. Consider:")
            print("   - Reducing batch size")
            print("   - Enabling gradient accumulation")
            print("   - Using a smaller model")
        elif total_memory_gb < 40:
            print("✅ Moderate GPU memory available. For larger models:")
            print("   - Consider gradient checkpointing")
            print("   - Use mixed precision training")
        else:
            print("✅ Substantial GPU memory available. You can use:")
            print("   - Larger batch sizes")
            print("   - Larger models")

if __name__ == "__main__":
    main()
