# 00_jax_vs_pytorch_philosophy.py

# Welcome to Module 0 of the JAX Learning Curriculum!
# Objective: Understand the fundamental design differences between JAX and PyTorch,
# focusing on JAX's functional programming core versus PyTorch's imperative style.

# --- Core Concepts ---
# 1. Functional Programming (FP) in JAX:
#    - Pure Functions: The ideal in JAX. Given the same inputs, a pure function
#      ALWAYS returns the same output and has NO side effects (like modifying
#      global variables, printing to console, changing input variables in-place).
#    - Immutability: JAX arrays, like Python tuples, are immutable. Operations
#      create *new* arrays instead of modifying existing ones.
#    - Function Transformations: JAX's power comes from transforming standard
#      Python/NumPy-like functions (e.g., `jit`, `grad`, `vmap`, `pmap`). These
#      transformations take functions as input and return new, transformed functions.
#
# 2. Imperative/Object-Oriented Style in PyTorch:
#    - Stateful Objects: PyTorch code often relies on objects (like `nn.Module`,
#      `optim.Optimizer`) that hold internal state (weights, optimizer moments).
#    - In-place Operations: PyTorch allows and often encourages in-place modification
#      of tensors (e.g., `tensor.add_()`, `tensor.relu_()`) for memory efficiency,
#      which introduces side effects.
#    - Execution Model: Primarily eager execution (operations run immediately),
#      though compilation (`torch.jit.script/trace`, `torch.compile`) exists.
#      Autograd (`.backward()`) is called explicitly to compute gradients, which
#      are then stored as attributes (`.grad`) on the tensors themselves (another side effect).

# Let's illustrate with simple examples.



# === Example 1: Simple Arithmetic and Side Effects ===

# -- PyTorch: In-place modification (Side Effect) --
import torch

# Create a tensor
pt_tensor = torch.tensor([1.0, 2.0, 3.0])
print(f"PyTorch Original Tensor: {pt_tensor}")

# In-place addition (modifies the original tensor)
pt_tensor.add_(10.0) # The underscore signifies in-place
print(f"PyTorch Tensor after in-place add_: {pt_tensor}")

# Standard addition (creates a new tensor)
pt_tensor_new = pt_tensor + 5.0
print(f"PyTorch Tensor after standard add (new tensor): {pt_tensor_new}")
print(f"PyTorch Original Tensor (unchanged by standard add): {pt_tensor}") # Note: pt_tensor still holds the value after add_

print("\n" + "-"*30 + "\n")


# -- JAX: Immutability (No Side Effects from Operations) --
import jax
import jax.numpy as jnp # JAX's NumPy-compatible API

# Create a JAX array
jax_array = jnp.array([1.0, 2.0, 3.0])
print(f"JAX Original Array: {jax_array}")

# Attempting an operation 'in-place' style doesn't exist or work like PyTorch.
# We always create new arrays.
jax_array_new = jax_array + 10.0
print(f"JAX Array after addition (new array): {jax_array_new}")
print(f"JAX Original Array (unchanged): {jax_array}") # The original is unaffected

# Key Takeaway: JAX operations produce new values rather than modifying existing ones.
# This makes reasoning about code easier, especially when combined with transformations.

print("\n" + "="*30 + "\n")



# === Example 2: Function Purity ===

# -- Impure function (modifies external state) --
# This is common in standard Python but discouraged for functions JAX transforms.
global_counter_impure = 0

def impure_function(x):
    global global_counter_impure
    global_counter_impure += 1 # Side effect: modifies global state
    print(f"Inside impure_function, counter: {global_counter_impure}") # Side effect: I/O
    return x * 2 # The calculation itself might be deterministic

print("Calling impure_function:")
result1_impure = impure_function(5)
result2_impure = impure_function(5)
print(f"Impure Results (same input, potentially different surrounding state): {result1_impure}, {result2_impure}")
print(f"Final global counter: {global_counter_impure}")
print("\n" + "-"*30 + "\n")


# -- Pure function (JAX style) --
# To achieve similar functionality in a pure way, state must be handled explicitly.
def pure_function_with_state(counter_in, x):
    # No global variables modified, no printing inside (ideally)
    counter_out = counter_in + 1 # Calculate new state
    result = x * 2               # Calculate result
    return result, counter_out   # Return both result and new state

print("Calling pure_function_with_state:")
current_counter = 0
result1_pure, current_counter = pure_function_with_state(current_counter, 5)
result2_pure, current_counter = pure_function_with_state(current_counter, 5)
print(f"Pure Results: {result1_pure}, {result2_pure}")
print(f"Final counter state (managed explicitly): {current_counter}")

# Key Takeaway: Pure functions make behavior predictable. JAX transformations
# like `jax.jit` often assume purity. When JIT-compiling, side effects like
# printing might happen only once during tracing, not every time the function
# runs, leading to confusion if you rely on them. Explicit state management
# (passing state in and getting new state out) is the JAX way.

print("\n" + "="*30 + "\n")



# === Example 3: Function Transformations (Conceptual Introduction) ===

# JAX provides functions that take your functions and return new, enhanced functions.

# -- Hypothetical Python function --
def my_python_function(x):
    return x * x + 2 * x + 1

# -- JAX transformations (we'll learn these properly later) --

# 1. `jax.jit`: Compiles the function for speed (often on GPU/TPU)
#    Takes `my_python_function`, returns a compiled version.
jitted_function = jax.jit(my_python_function)
# `jitted_function` behaves like `my_python_function` but might run much faster.

# 2. `jax.grad`: Creates a function that computes the gradient.
#    Takes `my_python_function`, returns a function that computes its derivative.
grad_function = jax.grad(my_python_function)
# `grad_function(3.0)` would compute the derivative of `my_python_function` evaluated at x=3.0.

# 3. `jax.vmap`: Creates a function that automatically handles batches (vectorization).
#    Takes a function designed for a single example, returns one that works on batches.
#    Imagine `my_python_function` only worked on scalars. `vmap` would make it work on arrays element-wise.
batched_function = jax.vmap(my_python_function)
# `batched_function(jnp.array([1.0, 2.0, 3.0]))` would apply the function to each element.


# -- Contrast with PyTorch --
# In PyTorch, these capabilities are often achieved differently:
# - Compilation: `torch.compile()` (newer) or `torch.jit.script/trace` can compile models/functions.
# - Gradients: Achieved via `tensor.requires_grad=True`, running ops, calling `loss.backward()`,
#   and accessing the `.grad` attribute of tensors. It's more integrated with the tensor objects.
# - Batching: Usually handled by designing layers (`nn.Linear`, `nn.Conv2d`, etc.) to expect
#   a batch dimension from the start. `vmap` provides more flexibility to vectorize arbitrary functions.

# Key Takeaway: JAX revolves around writing simple, pure Python/NumPy-like functions
# and then transforming them using `jit`, `grad`, `vmap`, `pmap`, etc., to add
# performance, differentiability, vectorization, and parallelism. PyTorch integrates
# these features more tightly with its Tensor object and `nn.Module` class structure.

print("\nModule 0 Summary:")
print(" - JAX favors pure functions and immutability.")
print(" - PyTorch uses stateful objects and allows in-place operations (side effects).")
print(" - JAX enhances functions via transformations (`jit`, `grad`, `vmap`).")
print(" - PyTorch integrates features like autograd directly into Tensor objects.")
print(" - Understanding purity and explicit state handling is crucial for JAX.")

# End of Module 0