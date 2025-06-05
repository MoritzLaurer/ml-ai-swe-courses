# 02_automatic_differentiation.py

# Welcome to Module 2 of the JAX Learning Curriculum!
# Objective: Learn how to compute gradients using `jax.grad` and related functions.
# Theme Integration: We'll define a simple loss function for our linear layer
#                    and compute gradients of the loss with respect to the layer's parameters (W, b).

import jax
import jax.numpy as jnp
import numpy as np
import time



# --- 1. Recap: Linear Layer Forward Pass ---
# Let's reuse the setup from Module 1.

# Layer parameters (dummy values)
input_dim = 4
output_dim = 2
# Use a JAX key for consistent initialization if needed
key = jax.random.PRNGKey(0) # Use a fixed seed for reproducibility here
key, W_key, b_key = jax.random.split(key, 3)

W = jax.random.normal(W_key, (input_dim, output_dim), dtype=jnp.float32)
b = jax.random.normal(b_key, (output_dim,), dtype=jnp.float32)
print(f"W: {W}\nb: {b}")

# Forward pass function (Pure)
def linear_forward_pass(params, x):
  """Computes the forward pass of a linear layer using a params tuple."""
  W, b = params # Unpack parameters
  return jnp.matmul(x, W) + b

# Dummy input data (batch size 1)
dummy_input = jnp.ones((1, input_dim), dtype=jnp.float32)

# Group parameters into a tuple (this is a simple 'PyTree')
parameters = (W, b)

# Run forward pass
output = linear_forward_pass(parameters, dummy_input)
print(f"Parameters (W shape: {W.shape}, b shape: {b.shape})")
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Output value:\n{output}")

print("\n" + "="*30 + "\n")



# --- 2. Defining a Loss Function ---
# To compute gradients, we need a scalar function to differentiate: the loss function.
# Let's use Mean Squared Error (MSE) between the output and a dummy target.

# Dummy target output
dummy_target = jnp.array([[0.5, -0.5]], dtype=jnp.float32) # Shape (1, output_dim)
print(f"Dummy Target shape: {dummy_target.shape}")
print(f"Dummy Target value:\n{dummy_target}")

# Loss function (MSE)
def mse_loss(params, x, y_target):
  """Calculates the Mean Squared Error loss."""
  y_pred = linear_forward_pass(params, x)
  # Calculate squared error per element, then average over the output dimensions and batch
  loss = jnp.mean(jnp.square(y_pred - y_target))
  return loss

# Calculate the loss for our dummy data
loss_value = mse_loss(parameters, dummy_input, dummy_target)
print(f"\nMSE Loss: {loss_value:.4f}")

# ** PyTorch Contrast **
# - Loss functions in PyTorch (e.g., `torch.nn.MSELoss`) are often classes (`nn.Module`)
#   that hold state (like reduction type), but the core calculation is similar.
# - You'd typically call `criterion = torch.nn.MSELoss()` and then `loss = criterion(output, target)`.

print("\n" + "="*30 + "\n")



# --- 3. Computing Gradients with `jax.grad` ---
# `jax.grad` transforms a function f(x, ...) into a function that computes the
# gradient of f with respect to one of its arguments.
# By default, it differentiates with respect to the *first* argument.

# Let's create a gradient function for our mse_loss w.r.t `params`.
# Since `params` is the first argument of `mse_loss`, this is the default.
grad_fn = jax.grad(mse_loss) # Default: Differentiates w.r.t. the first argument (params)

# Call the gradient function with the *same arguments* as the original loss function.
# The output will be the gradient(s) matching the structure of the differentiated argument.
gradients = grad_fn(parameters, dummy_input, dummy_target)

# `gradients` will have the same structure as `parameters` (a tuple of (grad_W, grad_b))
grad_W, grad_b = gradients

print(f"Gradients type: {type(gradients)}")
print(f"Gradient w.r.t. W (shape {grad_W.shape}):\n{grad_W}")
print(f"Gradient w.r.t. b (shape {grad_b.shape}):\n{grad_b}")

# ** Key Concept: Functional Nature **
# - `jax.grad(mse_loss)` didn't *run* anything yet; it returned a *new function* (`grad_fn`).
# - Calling `grad_fn(...)` executes the gradient computation.
# - The gradients are returned directly, not stored as attributes on the parameters.

# ** PyTorch Contrast **
# - PyTorch uses `loss.backward()`. This computes gradients for all tensors involved
#   in the loss computation graph that have `requires_grad=True`.
# - Gradients are stored in the `.grad` attribute of the respective tensors (e.g., `W.grad`).
#   This is a side effect / state modification.
# - You need to call `optimizer.zero_grad()` before `loss.backward()` in a training loop
#   to clear gradients from the previous step, as PyTorch accumulates them by default.
#   JAX doesn't accumulate gradients this way; each call to `grad_fn` computes fresh values.

print("\n" + "-"*30 + "\n")



# --- 4. Differentiating W.R.T. Other Arguments (`argnums`) ---
# What if `params` wasn't the first argument, or you wanted gradients w.r.t. the input `x`?
# Use the `argnums` argument for `jax.grad`.

def mse_loss_reordered(x, params, y_target): # Reordered arguments
  """Same MSE loss, but different argument order."""
  y_pred = linear_forward_pass(params, x)
  loss = jnp.mean(jnp.square(y_pred - y_target))
  return loss

# Differentiate w.r.t. `params` (which is now argument 1 - 0-indexed)
grad_fn_wrt_params = jax.grad(mse_loss_reordered, argnums=1)
gradients_params = grad_fn_wrt_params(dummy_input, parameters, dummy_target)
print(f"Gradients w.r.t params (argnums=1): Shapes ({gradients_params[0].shape}, {gradients_params[1].shape})")

# Differentiate w.r.t. `x` (argument 0)
grad_fn_wrt_x = jax.grad(mse_loss_reordered, argnums=0)
gradients_x = grad_fn_wrt_x(dummy_input, parameters, dummy_target)
print(f"Gradient w.r.t x (argnums=0): Shape ({gradients_x.shape})")

# Differentiate w.r.t. multiple arguments (returns a tuple of gradients)
grad_fn_multi = jax.grad(mse_loss_reordered, argnums=(0, 1))
gradients_multi = grad_fn_multi(dummy_input, parameters, dummy_target)
grad_x_multi, grad_params_multi = gradients_multi
print(f"Gradients w.r.t x and params (argnums=(0, 1)): Shapes ({grad_x_multi.shape}, ({grad_params_multi[0].shape}, {grad_params_multi[1].shape}))")

print("\n" + "="*30 + "\n")



# --- 5. Getting Value and Gradient Together (`jax.value_and_grad`) ---
# Often, you need both the loss value and the gradients (e.g., for logging the loss
# and updating parameters). Calling the loss function then the grad function is
# slightly inefficient as work is repeated.
# `jax.value_and_grad` computes both in one go.

# Use the original mse_loss where params is the first arg (argnums=0)
value_grad_fn = jax.value_and_grad(mse_loss, argnums=0) # Differentiate w.r.t params

# Calling this function returns a tuple: (value, gradients)
loss_value_again, gradients_again = value_grad_fn(parameters, dummy_input, dummy_target)

print(f"Using value_and_grad:")
print(f"Loss value: {loss_value_again:.4f}")
print(f"Gradient shapes: ({gradients_again[0].shape}, {gradients_again[1].shape})")

# This is a very common pattern in JAX training loops.

# ** PyTorch Contrast **
# - In PyTorch, you typically compute the loss value first (`loss = ...`), then call
#   `loss.backward()`. You access the loss value directly and the gradients via `.grad`.
#   There isn't a direct single function call that returns both detached loss and computes grads.

print("\n" + "="*30 + "\n")



# --- 6. Combining with JIT ---
# Gradient functions obtained from `jax.grad` or `jax.value_and_grad` are just
# regular functions that can themselves be JIT-compiled for performance.

jitted_value_grad_fn = jax.jit(jax.value_and_grad(mse_loss, argnums=0))

# First call (compiles)
start_time = time.time()
loss_val_jit, grad_jit = jitted_value_grad_fn(parameters, dummy_input, dummy_target)
end_time = time.time()
print("Running jitted value_and_grad (first call):")
print(f"Loss: {loss_val_jit:.4f}, Time: {end_time - start_time:.6f}s")

# Second call (uses cached compiled code)
start_time = time.time()
loss_val_jit_2, grad_jit_2 = jitted_value_grad_fn(parameters, dummy_input, dummy_target)
end_time = time.time()
print("\nRunning jitted value_and_grad (second call):")
print(f"Loss: {loss_val_jit_2:.4f}, Time: {end_time - start_time:.6f}s")

print("\n" + "="*30 + "\n")



# --- Module 2 Summary ---
# - `jax.grad(fun, argnums=...)` creates a new function that computes the gradient of `fun`
#   with respect to the argument(s) specified by `argnums` (default is 0, the first argument).
# - The returned gradient function is called with the same arguments as the original function.
# - The output of the gradient function matches the structure (shape and type) of the
#   argument(s) being differentiated.
# - `jax.value_and_grad(fun, ...)` is often more convenient and efficient, returning
#   both the function's output value and the computed gradients as a tuple `(value, grads)`.
# - JAX's gradient computation is functional: it takes a function and returns a gradient
#   function, without modifying state implicitly like PyTorch's `.backward()` and `.grad`.
# - JAX does not require manual gradient clearing (`zero_grad()`) because gradients are
#   returned directly, not accumulated in parameter attributes.
# - Gradient functions can be composed with `jax.jit` for performance.

# End of Module 2