# 01_jax_basics_and_jit.py

# Welcome to Module 1 of the JAX Learning Curriculum!
# Objective: Understand the basics of JAX arrays (`jax.Array`), the `jax.numpy` API,
#            the performance benefits of `jax.jit`, and JAX's explicit random number handling.
# Theme Integration: We'll define parameters for a simple linear layer and implement
#                    its forward pass using jax.numpy, then optimize it with jit.

import jax
import jax.numpy as jnp
import numpy as np # Often used alongside JAX, especially for initial data prep
import time



# --- 1. JAX Arrays and jax.numpy ---
# JAX provides its own array implementation (`jax.Array`) that is tightly integrated
# with its transformations (jit, grad, etc.) and accelerator backends (GPU, TPU).
# `jax.numpy` is the primary API for interacting with these arrays, designed to be
# very similar to NumPy.

# Creating JAX arrays:
jax_arr_a = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
jax_arr_b = jnp.arange(3, dtype=jnp.int32)

print(f"JAX Array A: {jax_arr_a}, Type: {type(jax_arr_a)}, Dtype: {jax_arr_a.dtype}")
print(f"JAX Array B: {jax_arr_b}, Type: {type(jax_arr_b)}, Dtype: {jax_arr_b.dtype}")

# Common operations using jax.numpy (look just like NumPy!)
jax_sum = jnp.add(jax_arr_a, 5.0) # Element-wise addition
jax_dot = jnp.dot(jax_arr_a, jax_arr_a) # Dot product
jax_matmul = jnp.matmul(jax_arr_a.reshape(1, 3), jax_arr_a.reshape(3, 1)) # Matrix multiplication

print(f"\nSum (A + 5.0): {jax_sum}")
print(f"Dot product (A . A): {jax_dot}")
print(f"Matrix multiplication: {jax_matmul}")

# Immutability: Remember from Module 0, JAX arrays are immutable.
# This operation creates a *new* array.
jax_arr_a_plus_10 = jax_arr_a + 10.0
print(f"\nOriginal JAX Array A still: {jax_arr_a}")
print(f"New array (A + 10.0): {jax_arr_a_plus_10}")

# ** PyTorch Contrast **
# - `jnp.array([...])` is analogous to `torch.tensor([...])`.
# - `jnp` functions (`jnp.add`, `jnp.dot`, etc.) mirror `torch` functions (`torch.add`, `torch.dot`).
# - Core difference: PyTorch tensors *can* be modified in-place (`tensor.add_()`),
#   while JAX arrays cannot. This immutability is key for JAX's functional nature.

print("\n" + "="*30 + "\n")



# --- 2. Theme Integration: Simple Linear Layer Forward Pass ---
# Let's define a simple linear layer: output = input @ W + b
# where '@' is matrix multiplication.

# Define layer parameters (as JAX arrays)
# Input features = 4, Output features = 2
input_dim = 4
output_dim = 2

# Usually, weights are learned, but here we'll just create some dummy ones.
# We'll use NumPy first to show interoperability, then convert to JAX arrays.
np_W = np.random.randn(input_dim, output_dim).astype(np.float32)
np_b = np.random.randn(output_dim).astype(np.float32)

# Convert NumPy arrays to JAX arrays
W = jnp.array(np_W)
b = jnp.array(np_b)
# Alternatively, create directly: W = jax.random.normal(key, (input_dim, output_dim))
# We'll cover jax.random properly later in this module.

print(f"Weight matrix W (shape {W.shape}):\n{W}")
print(f"Bias vector b (shape {b.shape}):\n{b}")

# Define the forward pass function (Pure Function)
def linear_forward_pass(x, W, b):
  """Computes the forward pass of a linear layer."""
  # Note: This function only uses its inputs to compute outputs.
  # It doesn't modify W, b, or x, and has no other side effects. It's pure.
  return jnp.matmul(x, W) + b

# Create some dummy input data (a single data point)
dummy_input = jnp.ones((1, input_dim), dtype=jnp.float32) # Shape (1, 4) for batch size 1
print(f"\nDummy Input (shape {dummy_input.shape}):\n{dummy_input}")

# Run the forward pass
output = linear_forward_pass(dummy_input, W, b)
print(f"\nOutput of linear_forward_pass (shape {output.shape}):\n{output}")

# ** PyTorch Contrast **
# - Defining parameters is similar (e.g., `torch.randn`).
# - The forward pass logic is almost identical using `torch.matmul` or `@`.
# - In PyTorch, `W` and `b` would typically be part of an `nn.Linear` module's state,
#   accessed via `self.weight` and `self.bias` within a `forward` method.
#   JAX encourages passing parameters explicitly into functions.

print("\n" + "="*30 + "\n")



# --- 3. Just-In-Time Compilation with `jax.jit` ---
# JAX can compile your Python functions using XLA (Accelerated Linear Algebra)
# for significant speedups, especially on GPUs and TPUs. `jax.jit` is the
# function transformation that does this.

# How it works (simplified):
# 1. Tracing: The first time you call the jitted function with specific input
#    shapes and dtypes, JAX traces its execution using abstract values (tracers).
#    It records the sequence of primitive operations.
# 2. Compilation: The traced sequence of operations is compiled into highly
#    optimized machine code (e.g., GPU kernels) using XLA.
# 3. Execution: Subsequent calls with the same input shapes/dtypes reuse the
#    compiled code directly, bypassing the Python interpreter overhead.

# Let's jit our forward pass function
jitted_linear_forward_pass = jax.jit(linear_forward_pass)

# First call: Tracing and compilation happen here. Might be slightly slower.
print("Running jitted function (first call - includes compilation time):")
start_time = time.time()
output_jitted_1 = jitted_linear_forward_pass(dummy_input, W, b)
end_time = time.time()
print(f"Output (jitted, 1st): {output_jitted_1}")
print(f"Time (1st call): {end_time - start_time:.6f} seconds")

# Second call: Uses the cached compiled code. Should be much faster (though
# the overhead might dominate for this *tiny* example; benefits are huge for complex functions).
print("\nRunning jitted function (second call - uses cached code):")
start_time = time.time()
output_jitted_2 = jitted_linear_forward_pass(dummy_input, W, b)
end_time = time.time()
print(f"Output (jitted, 2nd): {output_jitted_2}")
print(f"Time (2nd call): {end_time - start_time:.6f} seconds")

# JIT requires functions to be pure! Side effects (like printing inside the function)
# might behave unexpectedly (e.g., print only during tracing, not every execution).
@jax.jit
def problematic_jit_function(x):
  print("Inside problematic_jit_function! This might print only once.") # Side effect!
  return x * 2

print("\nTesting JIT with side effect:")
problematic_jit_function(5.0) # Print likely occurs here during trace
problematic_jit_function(6.0) # Print likely *doesn't* occur here

# ** PyTorch Contrast **
# - PyTorch primarily uses eager execution (ops run one by one).
# - `torch.jit.script` or `torch.jit.trace` were earlier ways to compile.
# - `torch.compile` (introduced in PyTorch 2.0) is the modern way, offering
#   significant speedups similar to JAX's JIT, using backends like Triton.
# - JAX's `jit` is arguably more central to the standard workflow from the start.
# - Purity constraints are more fundamental in JAX due to its functional design.

print("\n" + "="*30 + "\n")



# --- 4. Explicit Random Number Handling with `jax.random` ---
# Reproducibility is critical in ML. Most libraries (NumPy, PyTorch) use a global
# random number generator (RNG) state, seeded once (`np.random.seed()`, `torch.manual_seed()`).
# This global state is a side effect, which clashes with JAX's pure function philosophy.

# JAX requires explicit RNG state management via 'keys'.
# 1. Create an initial key: Usually seeded once at the start.
seed = 42
key = jax.random.PRNGKey(seed)
print(f"Initial PRNG Key: {key}")

# 2. Use the key for random operations: Functions like `jax.random.normal` require a key.
random_vector = jax.random.normal(key, (3,))
print(f"Random vector generated with key: {random_vector}")

# !!! CRITICAL: Using the same key twice produces the SAME output !!!
random_vector_same = jax.random.normal(key, (3,))
print(f"Same key, same output: {random_vector_same}")

# 3. Split the key: To get new random numbers, you must generate *new* keys
#    by splitting the old one. This ensures reproducibility and functional purity.
key, subkey = jax.random.split(key) # Split key into two new independent keys

print(f"\nSplit key into: new key = {key}, subkey = {subkey}")

random_vector_new = jax.random.normal(subkey, (3,)) # Use the subkey
print(f"Random vector generated with subkey: {random_vector_new}")

# Generate another random vector using the main key (which was updated by split)
key, subkey2 = jax.random.split(key) # Split again
random_vector_new_2 = jax.random.uniform(subkey2, (3,))
print(f"Another random vector (uniform) with subkey2: {random_vector_new_2}")
print(f"Main key is now: {key}")

# Typical pattern in JAX functions needing randomness:
def function_needs_random(key, shape):
  key, subkey = jax.random.split(key)
  random_data = jax.random.normal(subkey, shape)
  # ... do something with random_data ...
  return random_data, key # Return data and the *new* state of the key

output_data, key = function_needs_random(key, (2, 2))
print(f"\nOutput from function_needs_random:\n{output_data}")
print(f"Key after function_needs_random: {key}")

# ** PyTorch Contrast **
# - `torch.manual_seed(seed)` sets a global seed.
# - Subsequent calls to `torch.randn()`, `torch.rand()`, etc., implicitly use and
#   update this global RNG state.
# - This is convenient but makes it harder to guarantee reproducibility across
#   different execution orders or parallel settings, and violates functional purity.
# - JAX's explicit key management forces you to be deliberate about randomness,
#   which aids reproducibility and fits the functional paradigm + transformations.

print("\n" + "="*30 + "\n")



# --- Module 1 Summary ---
# - JAX uses immutable `jax.Array` objects, manipulated via the `jax.numpy` API.
# - `jax.numpy` provides a familiar NumPy-like interface for numerical operations.
# - Functions intended for JAX transformations should ideally be *pure* (no side effects).
# - `jax.jit` compiles JAX functions using XLA for significant performance gains,
#   especially on accelerators and for complex computations. It requires pure functions.
# - JAX handles randomness explicitly using `jax.random.PRNGKey`. Keys must be passed
#   to random functions and `jax.random.split` must be used to generate new randomness.
# - This explicit state management (for RNG keys, and later for model parameters/optimizer
#   state) is a core difference from the more implicit/object-oriented state handling
#   common in PyTorch.

# End of Module 1