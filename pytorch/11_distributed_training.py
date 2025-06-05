# -*- coding: utf-8 -*-
# 11_distributed_training.py

# Module 11: Introduction to Distributed Training (torch.distributed)
#
# This script provides a conceptual introduction to distributed training in PyTorch,
# focusing on DistributedDataParallel (DDP) for multi-GPU or multi-node training.
#
# We will cover:
# 1. Why Distributed Training?
# 2. Core Concepts: World Size, Rank, Backend, Process Group.
# 3. `torch.distributed` Package: Initialization and cleanup.
# 4. DistributedDataParallel (DDP):
#    - How it works (data parallelism, gradient synchronization).
#    - Key modifications to a single-GPU training script.
# 5. Data Loading with `DistributedSampler`.
# 6. Launching Distributed Training Scripts (torchrun, spawn).
# 7. Considerations for logging, saving checkpoints, etc.
# 8. Brief mention of other strategies (FSDP, Model Parallelism).
# 9. Comparisons to JAX distributed training.
#
# NOTE: This module is primarily explanatory. Running distributed training
# usually requires specific launch commands (e.g., `torchrun`) and an
# environment with multiple GPUs or nodes. The code snippets illustrate
# the necessary changes you'd make to a standard training script.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import os # For environment variables

print("--- Module 11: Introduction to Distributed Training (torch.distributed) ---\n")
print(f"Using PyTorch version: {torch.__version__}")

# %% 1. Why Distributed Training?
print("\n--- 1. Why Distributed Training? ---\n")

print("Training large models (like LLMs) on massive datasets can be very time-consuming")
print("or even impossible on a single GPU due to memory limitations.")
print("Distributed training allows you to leverage multiple GPUs (on one or more machines)")
print("to address these challenges by:")
print("- Reducing Training Time: Parallelizing computation across multiple devices.")
print("- Training Larger Models: Distributing the model or data can overcome single-device memory limits.")
print("  (DDP primarily helps with data parallelism, FSDP helps more with model size).")


# %% 2. Core Concepts in `torch.distributed`
print("\n--- 2. Core Concepts in `torch.distributed` ---\n")

print("- World Size: The total number of processes participating in the distributed training job.")
print("  E.g., if training on 4 GPUs, world size is 4.")
print("- Rank: A unique identifier (integer from 0 to world_size-1) assigned to each process.")
print("  The process with rank 0 is often called the 'master' or 'main' process, typically")
print("  handling tasks like logging, saving checkpoints, etc.")
print("- Backend: The communication library used for inter-process communication.")
print("  Common backends: 'nccl' (for NVIDIA GPUs, highly recommended), 'gloo' (CPU/GPU, good for Ethernet).")
print("- Process Group: A group of processes that can communicate with each other.")
print("  The default group includes all processes (the 'world').")


# %% 3. `torch.distributed` Package: Initialization and Cleanup
print("\n--- 3. `torch.distributed` Package: Initialization and Cleanup ---\n")

print("To use distributed training, each process must first initialize the process group.")

def setup_distributed(backend='nccl'):
    """Initializes the distributed process group."""
    # These environment variables are typically set by the launch utility (e.g., torchrun)
    # For single-node multi-GPU, LOCAL_RANK is often used to determine the GPU for the current process.
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0)) # Global rank

    if world_size > 1: # Only initialize if actually running in a distributed setting
        # Initialize the process group
        # MASTER_ADDR and MASTER_PORT are needed for multi-node communication.
        # torchrun usually sets these. For single-node, they might not be strictly needed
        # if the default init_method (e.g., "env://") can infer them.
        # dist.init_process_group(backend=backend, init_method='env://')
        # OR more explicitly if env vars aren't perfectly set:
        # dist.init_process_group(
        #     backend=backend,
        #     rank=rank,
        #     world_size=world_size
        # )
        # The 'env://' method is generally preferred as it reads from environment variables.
        dist.init_process_group(backend=backend)
        
        # Set the CUDA device for this process
        # Each process will control one GPU
        torch.cuda.set_device(local_rank)
        
        print(f"Initialized distributed training: Rank {rank}/{world_size}, Local Rank {local_rank} on GPU {torch.cuda.current_device()}")
        return rank, world_size, local_rank
    else:
        print("Not in a distributed environment (world_size=1). Skipping distributed setup.")
        return 0, 1, 0 # Defaults for non-distributed run

def cleanup_distributed():
    """Cleans up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("Destroyed distributed process group.")

# --- Conceptual script structure for initialization ---
# rank, world_size, local_rank = setup_distributed()
# ... your training code ...
# cleanup_distributed()


# %% 4. DistributedDataParallel (DDP)
print("\n--- 4. DistributedDataParallel (DDP) ---\n")

print("`DistributedDataParallel` (DDP) is a common strategy for data parallelism.")
print("How DDP works conceptually:")
print("1. Model Replication: The model is replicated on each participating GPU (each process).")
print("2. Data Sharding: The training dataset is split, and each process receives a unique")
print("   portion of the data (a 'shard') for each batch. (Handled by `DistributedSampler`).")
print("3. Forward Pass: Each process performs a forward pass on its local model replica with its data shard.")
print("4. Gradient Calculation: Each process computes gradients locally.")
print("5. Gradient Synchronization: DDP automatically synchronizes (averages) gradients across all processes")
print("   during the backward pass using all-reduce operations.")
print("6. Optimizer Step: Each process updates its local model parameters identically because they all")
print("   started with the same parameters and received the same averaged gradients.")
print("   This keeps the model replicas consistent across all GPUs.")

# --- Key Modifications to Your Training Script for DDP ---
print("\nKey modifications to a single-GPU training script to use DDP:")

# (a) Initialize distributed environment (as shown in section 3)
# rank, world_size, local_rank = setup_distributed()
# current_gpu_id = local_rank # Or torch.cuda.current_device() after set_device

# (b) Wrap your model with DDP
# model = YourModelClass(...).to(current_gpu_id) # Move model to its assigned GPU
# if world_size > 1:
#     model = DDP(model, device_ids=[current_gpu_id], output_device=current_gpu_id)
#     # For LLMs, find_unused_parameters=True might be needed if parts of the model
#     # don't participate in the loss computation in some iterations (e.g. conditional logic).
#     # model = DDP(model, device_ids=[current_gpu_id], output_device=current_gpu_id, find_unused_parameters=True)

print("`DDP(model, device_ids=[gpu_id])`: `device_ids` tells DDP which GPU the model for this process lives on.")
print("It also specifies the input/output device for the DDP module itself.")


# %% 5. Data Loading with `DistributedSampler`
print("\n--- 5. Data Loading with `DistributedSampler` ---\n")

print("When using DDP, each process needs to work on a different subset of the data.")
print("`torch.utils.data.DistributedSampler` handles this automatically.")
print("It ensures that each process gets a non-overlapping part of the dataset.")

# --- Conceptual DataLoader setup ---
# train_dataset = YourDataset(...)
# if world_size > 1:
#     train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
# else:
#     train_sampler = None # Or use a RandomSampler for single GPU

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=per_gpu_batch_size, # This is batch size PER GPU
#     shuffle=(train_sampler is None), # Shuffle is handled by the sampler in DDP
#     num_workers=your_num_workers,
#     pin_memory=True,
#     sampler=train_sampler
# )

print("\nKey points for `DistributedSampler`:")
print("- `num_replicas=world_size`, `rank=rank`: Essential for the sampler to know its role.")
print("- `shuffle=True` in `DistributedSampler`: Shuffles data across all processes.")
print("- `shuffle=False` (or `train_sampler is None`) in `DataLoader`: Avoids conflicting shuffles.")
print("- Effective Batch Size: The total batch size across all GPUs will be `per_gpu_batch_size * world_size`.")
print("  Adjust your learning rate or batch size accordingly.")


# %% 6. Launching Distributed Training Scripts
print("\n--- 6. Launching Distributed Training Scripts ---\n")

print("PyTorch provides utilities to launch distributed training scripts:")
print("1. `torchrun` (Recommended, formerly `torch.distributed.launch`):")
print("   A command-line tool that spawns multiple processes and sets up environment variables")
print("   (`MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`, `LOCAL_RANK`).")
print("   Usage (single node, 4 GPUs):")
print("   `torchrun --standalone --nproc_per_node=4 your_script.py --your_script_args`")
print("   For multi-node training, `torchrun` requires more arguments like `--nnodes`, `--node_rank`,")
print("   `--master_addr`, `--master_port`.")

print("\n2. `torch.multiprocessing.spawn`:")
print("   A Python function that can spawn processes. Useful for more programmatic control or")
print("   if `torchrun` is not suitable. Requires more manual setup of arguments for each process.")
print("   Example structure:")
print("""
import torch.multiprocessing as mp

def main_worker_function(rank, world_size, args):
    # Setup distributed for this specific rank
    # dist.init_process_group(backend='nccl', init_method='tcp://some_ip:port', rank=rank, world_size=world_size)
    # torch.cuda.set_device(rank)
    # ... create model, DDP, dataloader with sampler ...
    # ... training loop ...
    # cleanup_distributed()

if __name__ == '__main__':
    world_size = torch.cuda.device_count() # e.g., 4
    # args = ... parse your arguments ...
    # mp.spawn(main_worker_function,
    #          args=(world_size, args),
    #          nprocs=world_size,
    #          join=True)
""")


# %% 7. Considerations for Logging, Saving/Loading Checkpoints, Evaluation
print("\n--- 7. Considerations for Logging, Saving/Loading Checkpoints, Evaluation ---\n")

print("In a distributed setup, some operations should only be performed by one process (usually rank 0):")
print("- Logging: To avoid cluttered logs from multiple processes writing simultaneously.")
print("  `if rank == 0: logger.info(...)`")
print("- Saving Checkpoints: To prevent race conditions and redundant saves.")
print("  `if rank == 0: torch.save({'model_state_dict': model.module.state_dict(), ...}, path)`")
print("  Note: When saving a DDP model, access the original model via `model.module.state_dict()`.")
print("- Loading Checkpoints: All processes should load the same checkpoint to start consistently.")
print("  The model should be created on each GPU, then `load_state_dict` applied, then wrapped in DDP.")
print("  A barrier `dist.barrier()` might be needed after loading on rank 0 and before other ranks proceed,")
print("  to ensure rank 0 has finished saving/loading a file that others might read.")
print("- Evaluation: Can be done on rank 0 with the full dataset, or each rank can evaluate its shard")
print("  and then results are aggregated (e.g., using `dist.all_reduce` for loss/metrics).")
print("  `DistributedSampler(..., shuffle=False)` is typically used for validation.")


# %% 8. Brief Mention of Other Strategies (Beyond Basic DDP)
print("\n--- 8. Brief Mention of Other Strategies (Beyond Basic DDP) ---\n")

print("While DDP is excellent for data parallelism, other strategies exist, especially for very large models:")
print("- Fully Sharded Data Parallel (FSDP):")
print("  Part of PyTorch (`torch.distributed.fsdp`). More advanced than DDP.")
print("  Shards model parameters, gradients, and optimizer states across ranks.")
print("  Significantly reduces per-GPU memory, allowing training of much larger models.")
print("  More complex to set up and use than DDP.")
print("- Tensor Parallelism / Pipeline Parallelism (Model Parallelism):")
print("  Splits the model itself across multiple GPUs. Layers or parts of layers run on different devices.")
print("  Often requires manual model modification or specialized libraries (e.g., Megatron-LM, DeepSpeed).")
print("\nThis module focuses on DDP as it's a common and effective starting point for distributed training.")
print("FSDP is a natural next step for scaling to larger models within the PyTorch ecosystem.")


# %% 9. JAX Comparison for Distributed Training
print("\n--- 9. JAX Comparison for Distributed Training ---\n")

print("JAX handles distributed training differently, often leveraging its functional nature:")
print("- `jax.pmap` (Parallel Map): A core transformation for data parallelism. It compiles a function")
print("  to run in parallel across multiple devices (GPUs/TPUs).")
print("  `pmap` handles data sharding and gradient synchronization implicitly based on how data is fed")
print("  and how collective operations (like `jax.lax.pmean` for averaging gradients) are used within the `pmap`ped function.")
print("- SPMD (Single Program, Multiple Data): `pmap` embodies the SPMD paradigm. The same JAX code runs on all devices,")
print("  but operates on different data shards.")
print("- Explicit PRNG Keys & State: As with single-device JAX, PRNG keys and model/optimizer states must be handled explicitly")
print("  and replicated/sharded appropriately for `pmap`.")
print("- Compiler Optimizations: The JAX compiler (XLA) can perform extensive optimizations for the distributed computation graph.")
print("- Mesh Parallelism: For more complex parallelism (combining data, operator, pipeline), JAX often uses concepts of device meshes")
print("  and partitioning annotations (e.g., with libraries like `jax.experimental.shard_map` or higher-level frameworks).")
print("\nPyTorch DDP offers a more object-oriented, imperative approach, while JAX's `pmap` is a functional transformation.")
print("Both are powerful but have different programming models.")


# %% Conclusion
print("\n--- Module 11 Summary ---\n")
print("Key Takeaways:")
print("- Distributed training scales model training to multiple GPUs/nodes.")
print("- Core concepts: world size, rank, backend, process group.")
print("- `torch.distributed.init_process_group()` sets up communication.")
print("- `DistributedDataParallel (DDP)` replicates the model and synchronizes gradients.")
print("- `DistributedSampler` is crucial for sharding data correctly in DDP.")
print("- `torchrun` is the recommended tool for launching DDP scripts.")
print("- Rank 0 often handles tasks like logging and checkpointing.")
print("- This was a basic introduction; FSDP and model parallelism are more advanced techniques for larger models.")

print("\nEnd of Module 11.")
