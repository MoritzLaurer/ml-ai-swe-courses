# 03_vectorization_vmap.py

# Welcome to Module 3 of the JAX Learning Curriculum!
# Objective: Understand how `jax.vmap` automatically vectorizes functions
#            to handle batches of data efficiently.
# Theme Integration: We'll adapt our linear layer and loss function to process
#                    batches of inputs using `vmap`.

import jax
import jax.numpy as jnp
import time



# --- 1. Recap: Single-Instance Functions ---
# Let's refine our functions from Module 1 & 2 to clearly operate on a single
# data point (e.g., input `x` has shape `(input_dim,)`).

# Layer parameters
key = jax.random.PRNGKey(1) # New key for reproducibility
input_dim = 4
output_dim = 2
key, W_key, b_key = jax.random.split(key, 3)
W = jax.random.normal(W_key, (input_dim, output_dim), dtype=jnp.float32)
b = jax.random.normal(b_key, (output_dim,), dtype=jnp.float32)
parameters = (W, b) # Group parameters (PyTree)

# Single-example forward pass function
def single_example_forward_pass(params, x_single):
  """Linear layer forward pass for a single input vector x_single."""
  # Expects x_single to have shape (input_dim,)
  assert x_single.ndim == 1 and x_single.shape[0] == input_dim
  W_p, b_p = params
  return jnp.matmul(x_single, W_p) + b_p

# Single-example loss function (MSE)
def single_example_mse_loss(params, x_single, y_target_single):
  """MSE loss for a single input vector x_single and target y_target_single."""
  assert x_single.ndim == 1 and y_target_single.ndim == 1
  y_pred_single = single_example_forward_pass(params, x_single)
  loss = jnp.mean(jnp.square(y_pred_single - y_target_single))
  return loss

# Create single dummy input and target
key, x_key, y_key = jax.random.split(key, 3)
dummy_x_single = jax.random.normal(x_key, (input_dim,))
dummy_y_target_single = jax.random.normal(y_key, (output_dim,))

print(f"Single x shape: {dummy_x_single.shape}", f"Single x: {dummy_x_single}")
print(f"Single y_target shape: {dummy_y_target_single.shape}", f"Single y_target: {dummy_y_target_single}")

# Test the single-example functions
output_single = single_example_forward_pass(parameters, dummy_x_single)
loss_single = single_example_mse_loss(parameters, dummy_x_single, dummy_y_target_single)
print(f"\nOutput (single): {output_single}, shape: {output_single.shape}")
print(f"Loss (single): {loss_single:.4f}")

# ** The Problem: Handling Batches **
# What if we have a batch of inputs? e.g., `batch_x` with shape `(batch_size, input_dim)`
# We could loop in Python, but that's slow. We could rewrite the functions to handle
# batch dimensions manually (e.g., `jnp.matmul` handles this naturally if shapes align),
# but `vmap` provides a more general and often simpler way.

print("\n" + "="*30 + "\n")



# --- 2. Automatic Vectorization with `jax.vmap` ---
# `jax.vmap` transforms a function written for single examples into one that processes batches.
# Key argument: `in_axes` specifies which axis to map over for each input argument.
# `None` means the argument is *not* mapped over (e.g., parameters).
# `0` means map over the first axis (the conventional batch axis).
# `1` means map over the second axis, etc.

# Vectorize the forward pass
# We want to map over `x_single` (the 2nd arg, index 1), but NOT `params` (1st arg, index 0).
# So, `in_axes = (None, 0)`
batched_forward_pass = jax.vmap(
    single_example_forward_pass, # Function to vectorize
    in_axes=(None, 0)            # Do not map over params (first arg), map over the second arg (x) at axis 0
)

# Create a batch of dummy inputs
batch_size = 8
key, batch_x_key = jax.random.split(key, 2)
batch_x = jax.random.normal(batch_x_key, (batch_size, input_dim))
print(f"Batch x shape: {batch_x.shape}")

# Call the batched function
batch_output = batched_forward_pass(parameters, batch_x)
print(f"Batched output shape: {batch_output.shape}") # Should be (batch_size, output_dim)
# Note: We didn't change the logic inside `single_example_forward_pass`!

print("\n" + "-"*30 + "\n")



# --- 3. Vectorizing the Loss Function ---
# We want to compute the loss for each example in the batch.
# We need to map over `x_single` (arg 1) and `y_target_single` (arg 2).
# Parameters (arg 0) should remain the same for all examples.
# So, `in_axes = (None, 0, 0)`
batched_mse_loss_per_example = jax.vmap(
    single_example_mse_loss,
    in_axes=(None, 0, 0) # Don't map params, map axis 0 of x, map axis 0 of y_target
)

# Create a batch of dummy targets
key, batch_y_key = jax.random.split(key, 2)
batch_y_target = jax.random.normal(batch_y_key, (batch_size, output_dim))
print(f"Batch y_target shape: {batch_y_target.shape}")

# Compute loss for each example in the batch
batch_losses = batched_mse_loss_per_example(parameters, batch_x, batch_y_target)
print(f"Per-example losses shape: {batch_losses.shape}") # Should be (batch_size,)
print(f"Per-example losses:\n{batch_losses}")

# Often, we want the average loss over the batch for the training step.
def average_batch_mse_loss(params, x_batch, y_target_batch):
  """Calculates the average MSE loss over a batch using vmap internally."""
  # Use vmap to get per-example losses
  per_example_losses = jax.vmap(
      single_example_mse_loss, in_axes=(None, 0, 0)
  )(params, x_batch, y_target_batch)
  # Return the mean
  return jnp.mean(per_example_losses)

average_loss = average_batch_mse_loss(parameters, batch_x, batch_y_target)
print(f"\nAverage batch loss: {average_loss:.4f}")


# ** PyTorch Contrast **
# - PyTorch layers (`nn.Linear`, `nn.Conv2d`) are built assuming a batch dimension (usually axis 0).
#   The implementation handles the batching internally (e.g., matrix multiply expects [B, N] @ [N, M]).
# - If you write a custom PyTorch function, you often have to manually handle the batch dimension
#   using indexing, `torch.einsum`, or ensuring operations broadcast correctly.
# - `jax.vmap` allows writing clean logic for a single example and then automatically
#   vectorizing it, which can simplify code and prevent errors in manual batch handling.
# - PyTorch has `torch.vmap`, which provides similar functionality but is newer and perhaps
#   less commonly used than JAX's `vmap`, which is fundamental to the library.

print("\n" + "="*30 + "\n")



# --- 4. Composing `vmap` with `jit` and `grad` ---
# JAX transformations compose beautifully. We can jit the vmapped functions,
# or get gradients of the vmapped functions.

# Example: JIT the batched forward pass
jitted_batched_forward_pass = jax.jit(batched_forward_pass)

start_time = time.time()
output_jit = jitted_batched_forward_pass(parameters, batch_x)
end_time = time.time()
print("Running jitted vmapped forward pass (first call):")
print(f"Time: {end_time - start_time:.6f}s, Output shape: {output_jit.shape}")

start_time = time.time()
output_jit_2 = jitted_batched_forward_pass(parameters, batch_x)
end_time = time.time()
print("\nRunning jitted vmapped forward pass (second call):")
print(f"Time: {end_time - start_time:.6f}s, Output shape: {output_jit_2.shape}")


# Example: Get gradients of the *average* batch loss w.r.t parameters
value_grad_avg_batch_loss_fn = jax.value_and_grad(
    average_batch_mse_loss, # Use the function that calculates mean loss
    argnums=0               # Differentiate w.r.t. params (arg 0)
)

jitted_value_grad_avg_batch_loss = jax.jit(value_grad_avg_batch_loss_fn)

# Run the jitted value-and-gradient function for the batch
avg_loss_val, avg_loss_grads = jitted_value_grad_avg_batch_loss(parameters, batch_x, batch_y_target)
grad_W_avg, grad_b_avg = avg_loss_grads

print("\nJitted value_and_grad of average batch loss:")
print(f"Average Loss: {avg_loss_val:.4f}")
print(f"Gradient shapes: W={grad_W_avg.shape}, b={grad_b_avg.shape}")

# This `jitted_value_grad_avg_batch_loss` function is essentially the core
# computation needed inside a typical training step!

print("\n" + "="*30 + "\n")



# --- Module 3 Summary ---
# - `jax.vmap(fun, in_axes=...)` transforms a function `fun` written for single data points
#   into a function that operates over batches of data.
# - The `in_axes` argument is crucial: it's a tuple specifying which axis to map over for each
#   positional argument of `fun`. `None` indicates the argument should be broadcast (not mapped, e.g., parameters),
#   while `0` maps over the first dimension (typical batch axis).
# - `vmap` allows writing simpler, single-example logic and automatically handling batching,
#   often avoiding complex manual reshaping or indexing.
# - `vmap` composes seamlessly with other JAX transformations like `jit` and `grad`, enabling
#   the creation of efficient, batched, and differentiable functions from simple building blocks.
# - This contrasts with PyTorch where batch handling is often implemented explicitly within layer logic.

# End of Module 3