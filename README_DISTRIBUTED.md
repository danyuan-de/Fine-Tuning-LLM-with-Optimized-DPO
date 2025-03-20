# Distributed Training with DPO

This guide explains how to use the distributed training capabilities in this project to fine-tune language models using Direct Preference Optimization (DPO) across multiple GPUs.

## Prerequisites

- Multiple NVIDIA GPUs with CUDA support
- PyTorch 2.0.0 or higher
- NCCL (NVIDIA Collective Communications Library)
- Python 3.8+

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Check your system's capabilities for multi-GPU training:

```bash
python check_gpu.py
```

This will provide diagnostics about your GPUs and recommendations for distributed training.

## Running Distributed Training

### Using the Provided Script

The easiest way to start distributed training is by using the provided script:

```bash
chmod +x run_distributed.sh
./run_distributed.sh [num_gpus]
```

Where `[num_gpus]` is the number of GPUs to use (defaults to 2 if not specified).

### Manual Launch

You can also manually launch the distributed training:

```bash
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS src/main_distributed.py
```

Replace `NUM_GPUS` with the number of GPUs you want to use.

## Configuration

You can adjust the distributed training settings in `src/config.py`:

- `batch_size`: Will be automatically divided by the number of GPUs
- `learning_rate`: Will be automatically scaled based on the number of GPUs
- `gradient_accumulation_steps`: Increase for larger effective batch sizes without using more memory
- `use_gradient_checkpointing`: Enable to save memory at the cost of additional computation
- `use_mixed_precision`: Enable for faster training and reduced memory usage

## Monitoring Training

Training progress and metrics are automatically saved to the `results` directory:

- Loss and reward curves are plotted and saved as images
- Example outputs are saved to `distributed_output_val.txt` and `distributed_output_test.txt`
- The fine-tuned model is saved to the configured output directory

You can monitor GPU usage during training using:

```bash
nvidia-smi -l 1
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   - Reduce batch size
   - Enable gradient checkpointing
   - Enable mixed precision training
   - Increase gradient accumulation steps

2. **NCCL Timeout Errors**:
   - Set the environment variable: `export NCCL_DEBUG=INFO`
   - Set the environment variable: `export NCCL_SOCKET_IFNAME=eth0` (or your network interface)
   - Ensure all GPUs are on the same PCIe bus

3. **Process Group Initialization Failures**:
   - Ensure you have specified the correct world size and rank
   - Check that the master address and port are accessible
   - Make sure no other processes are using the same port

### Advanced Configuration

For more advanced distributed training options:

1. **Multi-Node Training**:
   - Set `MASTER_ADDR` to the IP address of the main node
   - Set `MASTER_PORT` to an accessible port
   - Set `NODE_RANK` to the rank of each node
   - Set `NNODES` to the total number of nodes

2. **Mixed Precision Training**:
   - Enable mixed precision in `config.py`
   - Uses FP16 or BF16 for computation while keeping model weights in FP32
   - Can significantly speed up training on GPUs with Tensor Cores (e.g., Volta, Turing, Ampere)

## Performance Tips

1. **Optimize Batch Size**:
   - Find the largest batch size that fits in GPU memory
   - Use gradient accumulation for larger effective batch sizes

2. **Learning Rate Scaling**:
   - Generally, scale the learning rate by the number of GPUs
   - May need further tuning for very large numbers of GPUs

3. **Data Loading**:
   - Use `num_workers` in DataLoader to parallelize data loading
   - Pin memory for faster CPU to GPU transfers

4. **GPU Placement**:
   - For multi-node setups, ensure fast network connectivity between nodes (InfiniBand preferred)
   - On a single node, use GPUs connected to the same PCIe switch for best performance
