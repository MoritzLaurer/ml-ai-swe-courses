# -*- coding: utf-8 -*-
# 02_tensors_and_autograd.py

# Module 2: Tensors, Autograd, and Computation Graphs in PyTorch
#
# This script dives into:
# 1. `torch.Tensor`: Creating, manipulating, and managing tensors (PyTorch's multi-dimensional arrays).
# 2. Device Placement: Moving tensors between CPU and GPU/MPS.
# 3. `torch.autograd`: PyTorch's automatic differentiation engine.
#    - How gradients are tracked (`requires_grad`).
#    - The dynamic computation graph.
#    - Calculating gradients (`.backward()`).
#    - Accessing computed gradients (`.grad`).
# 4. Disabling gradient tracking (`torch.no_grad`).

import torch
# No JAX needed explicitly here, but we'll add comments comparing concepts.
# import jax
# import jax.numpy as jnp

print("--- Module 2: Tensors, Autograd, and Computation Graphs ---\n")
print(f"Using PyTorch version: {torch.__version__}")


# %% 1. Creating Tensors
print("\n--- 1. Creating Tensors ---\n")

# From existing data (like Python lists or NumPy arrays)
data = [[1, 2], [3, 4]]
pt_from_list = torch.tensor(data) # Infers dtype (int64 here)
print(f"Tensor from list:\n{pt_from_list}")
print(f"  dtype: {pt_from_list.dtype}")

# Specify dtype (common types: torch.float32, torch.float64, torch.long, torch.int, torch.bool)
pt_float = torch.tensor(data, dtype=torch.float32)
print(f"Tensor with float32 dtype:\n{pt_float}")
print(f"  dtype: {pt_float.dtype}")

# From NumPy array (shares memory by default, unless NumPy array is copied)
import numpy as np
np_array = np.array(data)
pt_from_np = torch.from_numpy(np_array)
print(f"Tensor from NumPy array:\n{pt_from_np}")

# Using factory functions (like NumPy)
shape = (2, 3)
pt_ones = torch.ones(shape)
pt_zeros = torch.zeros(shape)
pt_randn = torch.randn(shape) # Samples from standard normal distribution
pt_rand = torch.rand(shape)   # Samples from uniform distribution [0, 1)
pt_arange = torch.arange(0, 10, 2) # Similar to range/np.arange
print(f"Tensor of ones:\n{pt_ones}")
print(f"Tensor of randn:\n{pt_randn}")
print(f"Tensor from arange: {pt_arange}")

# Getting Tensor attributes
print("\nAttributes of pt_randn:")
print(f"  Shape: {pt_randn.shape}")
print(f"  DataType: {pt_randn.dtype}") # Default floating point is float32
print(f"  Device: {pt_randn.device}") # Default device is CPU

# JAX Comparison:
# - `jnp.array()`, `jnp.ones()`, `jnp.zeros()`, etc. are similar.
# - JAX often defaults to 32-bit types. Check JAX docs for specific defaults.
# - JAX arrays are immutable.


# %% 2. Tensor Operations
print("\n--- 2. Tensor Operations ---\n")

# --- Indexing and Slicing (like NumPy) ---
tensor = torch.arange(12).reshape(3, 4)
print(f"Original Tensor:\n{tensor}")
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Element at (1, 1): {tensor[1, 1]}") # Returns a 0-dim tensor (scalar)
print(f"Get Python scalar value: {tensor[1, 1].item()}")
print(f"Last column: {tensor[:, -1]}")
print(f"Subgrid (rows 0-1, cols 1-2):\n{tensor[0:2, 1:3]}")

# Boolean indexing
bool_idx = tensor > 5
print(f"Boolean index (tensor > 5):\n{bool_idx}")
print(f"Elements > 5: {tensor[bool_idx]}") # Flattens the result

# --- Mathematical Operations ---
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.ones(2, 2) * 2

# Element-wise
print(f"Element-wise sum (a + b):\n{a + b}")
print(f"Element-wise product (a * b):\n{a * b}")
print(f"Element-wise sin(a):\n{torch.sin(a)}")

# Reduction
print(f"Sum of all elements in 'a': {a.sum()}") # or torch.sum(a)
print(f"Mean of all elements in 'a': {a.mean()}") # or torch.mean(a)
print(f"Sum across rows (dim=0): {a.sum(dim=0)}")
print(f"Sum across columns (dim=1): {a.sum(dim=1)}")

# Matrix Operations
print(f"Matrix multiplication (a @ b):\n{a @ b}") # or torch.matmul(a, b)
print(f"Matrix transpose (a.T):\n{a.T}") # or a.t() for 2D tensors

# In-place operations (modify tensor directly, usually end with '_')
# Use with caution, especially when autograd is involved.
c = torch.zeros(2, 2)
print("\nIn-place ops:")
print(f"Tensor 'c' before copy_:\n{c}")
c.copy_(a) # Copies data from 'a' into 'c'
print(f"Tensor 'c' after copy_(a):\n{c}")
c.add_(b) # c = c + b
print(f"Tensor 'c' after add_(b):\n{c}")

# --- Reshaping ---
d = torch.arange(8)
print("\nReshaping:")
print(f"Original tensor 'd': {d}")
# `.view()` requires tensor to be contiguous in memory, shares data
e_view = d.view(2, 4)
print(f"d.view(2, 4):\n{e_view}")
# `.reshape()` is more flexible, might return a copy if needed
e_reshape = d.reshape(2, 4)
print(f"d.reshape(2, 4):\n{e_reshape}")

# Add/remove dimensions
f = torch.randn(2, 3)
print(f"Original tensor 'f' shape: {f.shape}")
f_unsqueeze_0 = f.unsqueeze(0) # Add dim at index 0
print(f"f.unsqueeze(0) shape: {f_unsqueeze_0.shape}") # (1, 2, 3)
f_unsqueeze_1 = f.unsqueeze(1) # Add dim at index 1
print(f"f.unsqueeze(1) shape: {f_unsqueeze_1.shape}") # (2, 1, 3)
f_squeeze = f_unsqueeze_1.squeeze(1) # Remove dim at index 1 (if size 1)
print(f"f_unsqueeze_1.squeeze(1) shape: {f_squeeze.shape}") # (2, 3)

# Permute dimensions
g = torch.randn(2, 3, 4)
print(f"Original tensor 'g' shape: {g.shape}")
g_permute = g.permute(2, 0, 1) # Old dims (0, 1, 2) -> New dims (2, 0, 1)
print(f"g.permute(2, 0, 1) shape: {g_permute.shape}") # (4, 2, 3)

# JAX Comparison:
# - Operations API is very similar to `jnp`.
# - JAX favors `.reshape()` over `.view()`.
# - Key difference: JAX operations always return new arrays (immutability). No in-place ops.


# %% 3. Device Placement (CPU/GPU/MPS)
print("\n--- 3. Device Placement ---\n")

# Check for available accelerator devices
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA (GPU) is available. Using device: {device}")
elif torch.backends.mps.is_available(): # Metal Performance Shaders on Apple Silicon
    device = torch.device("mps")
    print(f"MPS is available. Using device: {device}")
else:
    device = torch.device("cpu")
    print(f"No GPU/MPS found. Using device: {device}")

# Create tensor on CPU then move
cpu_tensor = torch.randn(2, 2)
print(f"Tensor on CPU: {cpu_tensor.device}")
# Move tensor to the selected device
device_tensor = cpu_tensor.to(device)
print(f"Tensor moved to {device}: {device_tensor.device}")

# Create tensor directly on the device
direct_device_tensor = torch.randn(3, 3, device=device)
print(f"Tensor created directly on {device}: {direct_device_tensor.device}")

# Operations involving tensors on different devices will raise an error.
# Ensure tensors are on the same device before operating on them.
# e.g., result = device_tensor @ direct_device_tensor (This is OK)
# e.g., result = cpu_tensor @ device_tensor (This will ERROR)

# JAX Comparison:
# - Devices found via `jax.devices()`.
# - Explicit placement with `jax.device_put(x, device)`.
# - `jax.jit` often handles placement automatically based on where input arguments reside.


# %% 4. Autograd: Automatic Differentiation
print("\n--- 4. Autograd: Automatic Differentiation ---\n")

# --- The `requires_grad` Flag ---
# Tells PyTorch to track operations on this tensor for gradient computation.
x = torch.tensor([2.0], requires_grad=True) # MUST be float/complex dtype
w = torch.tensor([3.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)
print(f"x: {x}, requires_grad={x.requires_grad}")
print(f"w: {w}, requires_grad={w.requires_grad}")
print(f"b: {b}, requires_grad={b.requires_grad}")

# Operations involving tensors with requires_grad=True will produce outputs
# that also require gradients and have a `grad_fn`.
y = w * x + b # y = 3 * 2 + 1 = 7
print(f"y = w * x + b: {y}, requires_grad={y.requires_grad}")

# `grad_fn` points to the function that created the tensor (in the backward pass).
# It represents the backward operation in the computation graph.
print(f"Gradient function for y: {y.grad_fn}") # Should be <AddBackward0>

# You can stop tracking gradients using `.detach()`
z = y.detach()
print(f"z = y.detach(): {z}, requires_grad={z.requires_grad}") # False
print(f"z.grad_fn: {z.grad_fn}") # None

# --- Dynamic Computation Graph ---
# PyTorch builds a graph representing the computations *as they happen*.
# When `y = w * x + b` was executed:
# 1. `w * x` created an intermediate tensor with `grad_fn=<MulBackward0>`.
# 2. `(w * x) + b` created the final tensor `y` with `grad_fn=<AddBackward0>`.
# This graph stores the operations needed to compute gradients later.

# --- Calculating Gradients with `.backward()` ---
# To compute gradients, call `.backward()` on a SCALAR tensor (e.g., loss).
# It computes gradients of that scalar w.r.t. all tensors in the graph
# that have `requires_grad=True`.
# Let's compute d(y)/dx, d(y)/dw, d(y)/db
# y = w*x + b
# d(y)/dx = w = 3
# d(y)/dw = x = 2
# d(y)/db = 1

# Need to call backward on a scalar. 'y' is already scalar here.
print("\nCalculating gradients...")
y.backward(retain_graph=True) # Computes gradients and stores them in .grad attributes

# --- Accessing Gradients with `.grad` ---
print(f"Gradient dy/dx: {x.grad}") # Should be tensor([3.])
print(f"Gradient dy/dw: {w.grad}") # Should be tensor([2.])
print(f"Gradient dy/db: {b.grad}") # Should be tensor([1.])

# IMPORTANT: Understanding Gradient Accumulation & retain_graph
# By default, `.backward()` frees the computation graph after execution to save memory.
# Calling `.backward()` a second time on the *same output* (like 'y' here)
# without `retain_graph=True` on the first call will cause a RuntimeError.
# We used `retain_graph=True` in the first `y.backward()` call above specifically
# to keep the graph and allow a second backward pass for demonstration purposes.
# When the graph is retained, calling `backward()` again *will* accumulate (add)
# the newly computed gradients to the existing values in the `.grad` attribute.
# This is different from the typical gradient accumulation in training loops where 
# .backward() is called on different y from different batches.
# `.backward()` might be called multiple times (e.g., on different batches) *before*
# zeroing the gradients with `optimizer.zero_grad()`.
print("\nCalling backward again (possible because retain_graph=True was used)...")
y.backward() # Accumulate gradients (adds [3., 2., 1.] to existing grads)
print(f"Gradient dy/dx after second backward: {x.grad}") # Now tensor([6.])
print(f"Gradient dy/dw after second backward: {w.grad}") # Now tensor([4.])
print(f"Gradient dy/db after second backward: {b.grad}") # Now tensor([2.])

# In training loops, you MUST zero gradients before each *optimization step*.
# Typically done via `optimizer.zero_grad()` (more in Module 6).
# Manual way:
x.grad.zero_()
w.grad.zero_()
b.grad.zero_()
print(f"\nGradients after zeroing: dx={x.grad}, dw={w.grad}, db={b.grad}")

# Backward on non-scalar tensor requires specifying `gradient` argument
# (vector-Jacobian product)
v = torch.randn(3, requires_grad=True)
out = v * 2
# out.backward() # This would error!
print("\nBackward on non-scalar requires gradient argument:")
gradient_arg = torch.tensor([0.1, 1.0, 0.001])
out.backward(gradient=gradient_arg) # Computes vJP
print(f"Gradient d(out)/dv (vJP): {v.grad}") # Should be gradient_arg * 2

# JAX Comparison:
# - Gradients calculated via function transformation `jax.grad()` or `jax.value_and_grad()`.
# - No `.backward()` method or `.grad` attribute on arrays. `jax.grad` returns a function.
# - Computation graph is defined by the Python function and becomes static after `jax.jit`.


# %% 5. Disabling Gradient Tracking (`torch.no_grad()`)
print("\n--- 5. Disabling Gradient Tracking (`torch.no_grad()`) ---\n")

# Used during inference (when you don't need gradients) or when updating
# model parameters manually outside an optimizer. Saves memory and computation.
print(f"Original x: requires_grad={x.requires_grad}") # True
print(f"Original y: requires_grad={y.requires_grad}") # True

print("Inside torch.no_grad() context:")
with torch.no_grad():
    y_no_grad = w * x + b # Same computation
    print(f"  y_no_grad: {y_no_grad}, requires_grad={y_no_grad.requires_grad}") # False
    print(f"  y_no_grad.grad_fn: {y_no_grad.grad_fn}") # None
    # If we tried y_no_grad.backward(), it would raise an error.

# This is crucial during model evaluation (`model.eval()`) phase.

# JAX Comparison:
# - Not directly needed. Gradient calculation is opt-in via `jax.grad`.
# - Functions not wrapped in `jax.grad` don't compute gradients. Inference is just running the forward function.


# %% Conclusion
print("\n--- Module 2 Summary ---\n")
print("Key Takeaways:")
print("- `torch.Tensor` is the fundamental data structure, supporting NumPy-like operations.")
print("- Use `.to(device)` to manage CPU/GPU placement.")
print("- `torch.autograd` tracks operations on tensors with `requires_grad=True`.")
print("- A dynamic computation graph is built automatically.")
print("- `.backward()` computes gradients, stored in `.grad` attributes.")
print("- `with torch.no_grad():` disables gradient tracking for efficiency (e.g., inference).")

print("\nEnd of Module 2.")