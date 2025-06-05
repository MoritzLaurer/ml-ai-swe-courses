# -*- coding: utf-8 -*-
# Module 1: PyTorch vs. JAX - Core Concepts & Philosophy
#
# Welcome! This script compares the fundamental approaches of PyTorch and JAX.
# PyTorch is known for its imperative, define-by-run style, making it feel very
# Pythonic. JAX, built on functional programming principles, uses transformations
# like jit (just-in-time compilation), grad (autodiff), and vmap/pmap (vectorization/parallelization)
# on pure Python functions typically operating on NumPy-like arrays.
#
# Understanding these core differences will help you leverage your JAX knowledge
# while learning PyTorch.

# %% Importing the libraries
import torch
import jax
import jax.numpy as jnp
import numpy as np
import timeit

print("--- Module 1: PyTorch vs. JAX ---")
print(f"Using PyTorch version: {torch.__version__}")
print(f"Using JAX version: {jax.__version__}")



# %% 1. Tensor/Array Basics & API Style
# Both libraries provide NumPy-like objects for numerical computation.
# PyTorch uses `torch.Tensor`, JAX uses `jax.numpy.ndarray`.
# Their APIs are often very similar for basic operations.
print("\n--- 1. Tensor/Array Basics ---")

# PyTorch Tensor Creation
pt_x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
pt_y = torch.ones(2, 2) * 3.0
print(f"PyTorch Tensor:\n{pt_x}")

# JAX Array Creation
jax_x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
# Note: JAX tries to default to float32 or int32 unless specified otherwise
# For consistency with PyTorch default (float32), explicitly setting dtype often not needed
# but good practice if precision matters. Forcing float64 like numpy:
# jax_x_64 = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float64)
jax_y = jnp.ones((2, 2)) * 3.0
print(f"JAX Array:\n{jax_x}")

# Basic Operations (API similarity)
pt_z = pt_x + pt_y
pt_mm = torch.matmul(pt_x, pt_y)
print(f"PyTorch Addition:\n{pt_z}")
print(f"PyTorch Matmul:\n{pt_mm}")

jax_z = jax_x + jax_y
jax_mm = jnp.matmul(jax_x, jax_y)
# Important: JAX operations are typically asynchronous on accelerators.
# To ensure computation finishes for timing or printing, use .block_until_ready()
jax_mm.block_until_ready() # Ensure computation is done before printing
print(f"JAX Addition:\n{jax_z}")
print(f"JAX Matmul:\n{jax_mm}")

# Mutability vs. Immutability
# PyTorch tensors are mutable (like NumPy arrays), allowing in-place operations.
# JAX arrays are immutable. Operations always return new arrays.
print("\nMutability:")
pt_x_copy = pt_x.clone()
print(f"PyTorch tensor (original ID): {id(pt_x)}")
print(f"PyTorch tensor (.clone() ID): {id(pt_x_copy)}") # Different ID
pt_x_copy.add_(1.0) # In-place addition (note the trailing underscore '_')
print(f"PyTorch tensor (after inplace add_ on clone): ID {id(pt_x_copy)}\n{pt_x_copy}")
print(f"Original PyTorch tensor (unaffected): ID {id(pt_x)}\n{pt_x}") # Original is unchanged

# Simple assignment just creates another reference to the SAME tensor
pt_x_ref = pt_x
print("\nSimple assignment (pt_x_ref = pt_x):")
print(f"  Original pt_x ID: {id(pt_x)}")
print(f"  Reference pt_x_ref ID: {id(pt_x_ref)}") # Same ID as pt_x
pt_x_ref.mul_(2.0) # In-place multiplication via the reference
print(f"  Reference pt_x_ref after inplace mul_:\n{pt_x_ref}")
print(f"  Original pt_x is now also changed:\n{pt_x}") # Original IS changed

jax_x_copy = jnp.copy(jax_x)
print(f"JAX array (original ID): {id(jax_x_copy)}")
# JAX has no in-place operations like add_
try:
    # This would raise an error if such a method existed:
    # jax_x_copy.add_(1.0)
    # Instead, you reassign the result:
    jax_x_copy = jax_x_copy + 1.0
    print(f"JAX array (after functional add): ID {id(jax_x_copy)}\n{jax_x_copy}")
except AttributeError as e:
    print(f"JAX error as expected: {e}")
# Note the ID changed, indicating a new array was created.



# %% 2. Execution Model: Eager (Define-by-Run) vs. JIT Compilation (Define-then-Run)
# This is a major philosophical difference.
print("\n--- 2. Execution Model ---")

# PyTorch: Eager Execution
# Operations run immediately as they are called, like standard Python.
# This makes debugging intuitive (print statements, standard debuggers work directly).
print("PyTorch Eager Execution:")
pt_a = torch.randn(100, 100)
pt_b = torch.randn(100, 100)
print("  Performing PyTorch multiplication...")
start_time = timeit.default_timer()
pt_c = torch.matmul(pt_a, pt_b)
# The result pt_c is available immediately after the matmul line.
print("  PyTorch multiplication done.")
end_time = timeit.default_timer()
print(f"  PyTorch execution time: {end_time - start_time:.6f}s")
# Can directly print intermediate results or use debuggers easily
# print(pt_a) # Can inspect 'pt_a' right after its creation


# JAX: Functional Transformations (JIT)
# JAX operations can run eagerly like NumPy, but the real power comes from
# transformations like jax.jit(). `jit` traces a Python function and compiles
# it using XLA (Accelerated Linear Algebra) for high performance, especially on GPUs/TPUs.
# The Python code is run once during tracing to define the computation graph.
# Subsequent calls execute the optimized, compiled graph.
print("\nJAX Functional Programming & JIT:")

def jax_matmul_func(a, b):
    print("  ** TRACING JAX FUNCTION jax_matmul_func **") # Will print only on first call per shape/dtype
    c = jnp.matmul(a, b)
    # print(c) # Avoid side effects like printing inside JITted functions for performance
    return c

# Eager execution (like NumPy, useful for debugging)
jax_a = jnp.ones((100, 100))
jax_b = jnp.ones((100, 100))
print("  Performing JAX eager multiplication...")
start_time = timeit.default_timer()
jax_c_eager = jax_matmul_func(jax_a, jax_b)
jax_c_eager.block_until_ready() # Wait for computation
print("  JAX eager multiplication done.")
end_time = timeit.default_timer()
print(f"  JAX eager execution time: {end_time - start_time:.6f}s")


# JIT-compiled execution
jitted_matmul = jax.jit(jax_matmul_func)

print("\n  Performing JAX JITted multiplication (first call - includes compile time):")
start_time = timeit.default_timer()
jax_c_jitted1 = jitted_matmul(jax_a, jax_b)
jax_c_jitted1.block_until_ready() # Wait for computation
print("  JAX JITted multiplication (first call) done.")
end_time = timeit.default_timer()
print(f"  JAX JITted execution time (first): {end_time - start_time:.6f}s") # Includes compilation time

print("\n  Performing JAX JITted multiplication (second call - uses cache):")
start_time = timeit.default_timer()
jax_c_jitted2 = jitted_matmul(jax_a, jax_b) # Will likely reuse compiled function
jax_c_jitted2.block_until_ready() # Wait for computation
print("  JAX JITted multiplication (second call) done.")
end_time = timeit.default_timer()
print(f"  JAX JITted execution time (second): {end_time - start_time:.6f}s") # Should be much faster

# JIT makes debugging harder as the Python code isn't executed directly after the first trace.
# Errors might occur within the compiled XLA code. `jax.debug.print` or `host_callback`
# are needed for visibility inside JITted functions.



# %% 3. Statefulness vs. Statelessness (Conceptual Intro)
# This relates to how models and their parameters are handled.
print("\n--- 3. Statefulness vs. Statelessness ---")

# PyTorch: Stateful Modules
# `torch.nn.Module` objects encapsulate parameters (weights, biases) and code (forward pass).
# The parameters are part of the module's state. When you call `model(input)`, it uses its internal parameters.
# We will explore this in detail in Module 3.
print("PyTorch uses stateful `nn.Module` objects (more in Module 3).")
# Example sketch (not runnable yet):
# model = torch.nn.Linear(10, 2) # Creates layer with internal weight & bias state
# output = model(input_tensor)   # Uses internal state implicitly

# JAX: Stateless Functions
# JAX functions are typically "pure" - their output depends only on their explicit inputs.
# Model parameters are usually stored externally (e.g., in a dictionary or PyTree) and
# explicitly passed into the function during each call.
# Frameworks like Flax or Haiku help manage this state.
print("JAX typically uses stateless functions with explicit parameter passing.")
# Example sketch (not runnable yet):
# params = initialize_linear_params(key, 10, 2) # e.g., returns {'weight': ..., 'bias': ...}
# def linear_apply(params, input_tensor):
#     return jnp.dot(input_tensor, params['weight']) + params['bias']
# output = linear_apply(params, input_tensor) # State (params) passed explicitly



# %% 4. Automatic Differentiation (Autograd)
# Both provide ways to compute gradients, but the mechanism feels different.
print("\n--- 4. Automatic Differentiation ---")

# PyTorch: Dynamic Graph & `.backward()`
# PyTorch builds a computation graph dynamically as operations execute.
# Gradients are computed by calling `.backward()` on a scalar output tensor.
# Gradients are stored in the `.grad` attribute of the input tensors.
print("PyTorch Autograd:")
pt_x_grad = torch.tensor([2.0], requires_grad=True) # Need to track gradients for this tensor
pt_y_grad = pt_x_grad ** 3 # y = x^3 => dy/dx = 3*x^2
print(f"  PyTorch: y = {pt_y_grad.item()}")
pt_y_grad.backward() # Compute gradients w.r.t. inputs with requires_grad=True
print(f"  PyTorch: dy/dx at x=2.0 is: {pt_x_grad.grad.item()}") # Should be 3 * (2^2) = 12

# JAX: Functional Transformation `jax.grad()`
# JAX uses function transformations. `jax.grad()` takes a function and returns
# a new function that computes the gradient of the original function.
print("\nJAX Autograd:")
def cube_func(x):
  return jnp.sum(x ** 3) # grad requires scalar output

# Create the gradient function
grad_func = jax.grad(cube_func)

jax_x_grad = jnp.array([2.0])
jax_dy_dx = grad_func(jax_x_grad) # Call the gradient function
print(f"  JAX: y = {cube_func(jax_x_grad)}")
print(f"  JAX: dy/dx at x=2.0 is: {jax_dy_dx[0]}") # Should also be 12

# `jax.value_and_grad` is often used to get both the function's value and gradient simultaneously.
value_and_grad_func = jax.value_and_grad(cube_func)
jax_y_val, jax_dy_dx_val = value_and_grad_func(jax_x_grad)
print(f"  JAX: value_and_grad -> value: {jax_y_val}, grad: {jax_dy_dx_val[0]}")



# %% 5. Device Handling
# How tensors/arrays are moved between CPU, GPU, TPU.

print("\n--- 5. Device Handling ---")

# PyTorch: `.to()` method
print("PyTorch Device Handling:")
if torch.cuda.is_available():
    pt_device = torch.device("cuda")
    print(f"  PyTorch CUDA device found: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available(): # For Apple Silicon GPUs
    pt_device = torch.device("mps")
    print("  PyTorch MPS device found.")
else:
    pt_device = torch.device("cpu")
    print("  PyTorch using CPU.")

pt_t = torch.randn(2, 2)
print(f"  Tensor on CPU: {pt_t.device}")
pt_t_gpu = pt_t.to(pt_device)
print(f"  Tensor moved via .to(): {pt_t_gpu.device}")
# Models are also moved using `.to(device)` (more in Module 3)
# model.to(pt_device)


# JAX: `jax.devices()`, `jax.device_put()`, often implicit with JIT
print("\nJAX Device Handling:")
jax_devices = jax.devices()
print(f"  JAX available devices: {jax_devices}")

# JAX often places arrays automatically, especially inputs to JITted functions.
# Explicit placement can be done with `device_put`.
jax_a = jnp.array([1.0, 2.0])
print(f"  JAX array default device: {jax_a.device}") # Often CPU initially or based on context

try:
    # Try placing on the first GPU if available
    gpu_device = jax.devices('gpu')[0]
    jax_a_gpu = jax.device_put(jax_a, gpu_device)
    print(f"  JAX array moved via device_put to GPU: {jax_a_gpu.device}")
except (IndexError, RuntimeError): # RuntimeError if no GPU backend configured
    print("  JAX GPU not found or configured, using default device.")
    try:
        # Try placing on the first TPU if available
        tpu_device = jax.devices('tpu')[0]
        jax_a_tpu = jax.device_put(jax_a, tpu_device)
        print(f"  JAX array moved via device_put to TPU: {jax_a_tpu.device}")
    except (IndexError, RuntimeError):
      print("  JAX TPU not found or configured, using default device.")

# Often, you place initial data, and JIT compilation handles subsequent placement.



# %% Conclusion
print("\n--- Module 1 Summary ---")
print("Key Takeaways:")
print("- PyTorch: Imperative (eager), stateful (`nn.Module`), dynamic graph (`.backward()`), explicit `.to(device)`.")
print("- JAX: Functional (JIT compilation), stateless functions (explicit params), function transforms (`jax.grad`), often implicit device placement with JIT.")
print("Both offer powerful GPU/TPU acceleration and NumPy-like tensor operations.")
print("Understanding these differences helps in translating concepts between the two frameworks.")

print("\nEnd of Module 1.")