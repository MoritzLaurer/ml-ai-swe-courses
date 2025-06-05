# 04_combining_transforms_pytrees.py

# Welcome to Module 4 of the JAX Learning Curriculum!
# Objective: Learn how JAX handles nested data structures (PyTrees) containing
#            parameters, and how transformations compose and operate on them.
# Theme Integration: We'll define a simple 2-layer MLP using a nested dictionary
#                    (a PyTree) for parameters and apply combined transformations.

import jax
import jax.numpy as jnp
import jax.random as random # Use jax.random directly
import time



# --- 4.1 What are PyTrees? ---
# A PyTree is any nested structure built from Python containers like tuples,
# lists, and dictionaries, where the 'leaves' are typically JAX arrays (but can
# also be other objects JAX knows how to handle, or even non-JAX objects which
# are treated as static).
# JAX transformations operate on the leaves while preserving the container structure.

print("--- 4.1 PyTree Examples ---")

# Example 1: Tuple PyTree
pytree_tuple = (jnp.ones((2, 2)), jnp.zeros(3))
print(f"Tuple PyTree: {pytree_tuple}")

# Example 2: Dictionary PyTree
pytree_dict = {'layer1': jnp.arange(4.0), 'config': None, 'layer2': jnp.eye(2)}
print(f"Dictionary PyTree: {pytree_dict}")

# Example 3: Nested PyTree
pytree_nested = [{'a': (jnp.array(1), jnp.array(2))}, [jnp.array([3, 4])]]
print(f"Nested PyTree: {pytree_nested}")

# JAX provides utilities to work with PyTrees, e.g., `jax.tree.map`
# applies a function to each leaf.
shapes_tuple = jax.tree.map(lambda x: x.shape, pytree_tuple)
print(f"\nShapes of tuple PyTree leaves: {shapes_tuple}")

shapes_dict = jax.tree.map(lambda x: x.shape if hasattr(x, 'shape') else type(x), pytree_dict)
print(f"Shapes/Types of dict PyTree leaves: {shapes_dict}")

print("\n" + "="*30 + "\n")



# --- 4.2 Theme: Simple 2-Layer MLP Parameters as a PyTree ---
# Let's define parameters for an MLP: Linear(input->hidden) -> ReLU -> Linear(hidden->output)
# We'll store them in a nested dictionary.

input_dim = 4
hidden_dim = 8
output_dim = 2

key = random.PRNGKey(42) # Consistent key

def init_mlp_params(key, input_d, hidden_d, output_d):
  """Initializes parameters for a 2-layer MLP using a PyTree dict."""
  key, key_l1_w, key_l1_b, key_l2_w, key_l2_b = random.split(key, 5)

  # Glorot/Xavier initialization is common, let's use simple random normal for now
  scale1 = jnp.sqrt(2.0 / input_d)
  scale2 = jnp.sqrt(2.0 / hidden_d)

  params = {
      'linear1': {
          'W': random.normal(key_l1_w, (input_d, hidden_d)) * scale1,
          'b': random.normal(key_l1_b, (hidden_d,)) * scale1 # Biases often start near zero
      },
      'linear2': {
          'W': random.normal(key_l2_w, (hidden_d, output_d)) * scale2,
          'b': random.normal(key_l2_b, (output_d,)) * scale2
      }
  }
  return params

mlp_params = init_mlp_params(key, input_dim, hidden_dim, output_dim)

print("--- 4.2 MLP Parameters (PyTree Dictionary) ---")
# Print the structure with shapes
param_shapes = jax.tree.map(lambda x: x.shape, mlp_params)
print(f"Parameter shapes PyTree:\n{param_shapes}")

# Print some actual values (e.g., first weight matrix)
print(f"\nExample: mlp_params['linear1']['W'] (first 2 rows):\n{mlp_params['linear1']['W'][:2, :]}")


print("\n" + "="*30 + "\n")




# --- 4.3 MLP Forward Pass with PyTree Parameters ---
# Functions operating on PyTrees just access elements normally.

def forward_pass_mlp(params, x):
  """Performs a forward pass for the 2-layer MLP."""
  # Layer 1
  l1_out = jnp.matmul(x, params['linear1']['W']) + params['linear1']['b']
  l1_act = jax.nn.relu(l1_out) # Apply ReLU activation
  # Layer 2
  l2_out = jnp.matmul(l1_act, params['linear2']['W']) + params['linear2']['b']
  return l2_out

# Create a dummy batch of inputs
batch_size = 8
key, x_key = random.split(key)
batch_x = random.normal(x_key, (batch_size, input_dim))

print("--- 4.3 MLP Forward Pass ---")
print(f"Input batch shape: {batch_x.shape}")

# Use vmap to apply the forward pass to the batch
# Note: The `forward_pass_mlp` itself assumes a single input vector or a batch
# that matmul can handle directly. Here we assume the latter for simplicity.
# If it were written for a single vector, we'd use vmap(..., in_axes=(None, 0)).
batch_output_mlp = forward_pass_mlp(mlp_params, batch_x)

print(f"\nOutput batch shape: {batch_output_mlp.shape}")
print(f"Example output (first element of batch):\n{batch_output_mlp[0]}")

print("\n" + "="*30 + "\n")



# --- 4.4 Gradients with PyTrees ---
# `jax.grad` (and `value_and_grad`) work seamlessly with PyTree parameters.
# The returned gradients PyTree mirrors the structure of the parameter PyTree.

def mse_loss_mlp(params, x_batch, y_target_batch):
  """Calculates average MSE loss for the MLP over a batch."""
  y_pred_batch = forward_pass_mlp(params, x_batch)
  # Calculate loss per example, then average
  loss = jnp.mean(jnp.square(y_pred_batch - y_target_batch))
  return loss

# Create dummy targets
key, y_key = random.split(key)
batch_y = random.normal(y_key, (batch_size, output_dim))

# Get value and gradients w.r.t. `mlp_params` (arg 0)
value_grad_mlp_fn = jax.value_and_grad(mse_loss_mlp, argnums=0)

loss_val_mlp, grads_mlp = value_grad_mlp_fn(mlp_params, batch_x, batch_y)

print("--- 4.4 Gradients of MLP Loss ---")
print(f"Target batch shape: {batch_y.shape}")
print(f"Calculated Loss: {loss_val_mlp:.4f}")

# Check the structure of the gradients PyTree
print("\nGradient PyTree structure (matches parameter structure):")
grad_shapes = jax.tree.map(lambda g: g.shape, grads_mlp)
print(grad_shapes)

# Print an example gradient value
print(f"\nExample gradient: grads_mlp['linear1']['W'] (first 2 rows):\n{grads_mlp['linear1']['W'][:2, :]}")

print("\n" + "="*30 + "\n")



# --- 4.5 Composing Transformations with PyTrees ---
# We can easily combine `jit`, `vmap`, and `grad` on functions operating on PyTrees.

# Let's redefine the forward/loss slightly to be explicitly single-example
# for a clearer `vmap` demonstration.
def single_forward_mlp(params, x_single):
    assert x_single.ndim == 1
    l1_out = jnp.matmul(x_single, params['linear1']['W']) + params['linear1']['b']
    l1_act = jax.nn.relu(l1_out)
    l2_out = jnp.matmul(l1_act, params['linear2']['W']) + params['linear2']['b']
    return l2_out

def single_mse_loss_mlp(params, x_single, y_target_single):
    assert x_single.ndim == 1 and y_target_single.ndim == 1
    y_pred_single = single_forward_mlp(params, x_single)
    return jnp.mean(jnp.square(y_pred_single - y_target_single))

# Create the function that computes the average loss over a batch using vmap
def average_batch_loss_mlp_vmapped(params, x_batch, y_target_batch):
    # Use vmap to map over the batch dimension (0) of inputs and targets,
    # but not the parameters (None).
    per_example_losses = jax.vmap(
        single_mse_loss_mlp, in_axes=(None, 0, 0)
    )(params, x_batch, y_target_batch)
    return jnp.mean(per_example_losses)

# Now, get the value and gradient function for this *average batch loss*
value_grad_avg_batch_mlp_fn = jax.value_and_grad(average_batch_loss_mlp_vmapped, argnums=0)

# Finally, JIT-compile this value-and-gradient function
jitted_value_grad_avg_batch_mlp = jax.jit(value_grad_avg_batch_mlp_fn)

print("--- 4.5 Composing JIT, VMAP, GRAD with PyTrees ---")
print(f"Using JIT(VALUE_AND_GRAD(Vmapped average loss)) function.")

# Run the compiled function (first run includes compilation)
start_time = time.time()
loss_final, grads_final = jitted_value_grad_avg_batch_mlp(mlp_params, batch_x, batch_y)
end_time = time.time()
print(f"\nFirst run (compiled): Loss = {loss_final:.4f}, Time = {end_time - start_time:.6f}s")
print("Gradient structure:")
print(jax.tree.map(lambda g: g.shape, grads_final))

# Run again (should be faster)
start_time = time.time()
loss_final_2, grads_final_2 = jitted_value_grad_avg_batch_mlp(mlp_params, batch_x, batch_y)
end_time = time.time()
print(f"\nSecond run (cached): Loss = {loss_final_2:.4f}, Time = {end_time - start_time:.6f}s")

# ** PyTorch Contrast **
# - PyTorch `nn.Module` provides a class-based structure. Parameters are attributes
#   and submodules handle nested structures. `model.parameters()` or `model.state_dict()`
#   are used to access parameters, often yielding flattened lists or dictionaries.
# - JAX uses standard Python containers (tuples, lists, dicts) directly as PyTrees.
#   Transformations like `grad` naturally return gradients in the same nested structure.
# - Applying updates (Module 5 preview): In PyTorch, `optimizer.step()` modifies the
#   `.data` attribute of parameter tensors in-place. In JAX, we'll typically use
#   `jax.tree.map` within an update function to apply `param - lr * grad` to each leaf
#   (param, grad pair), creating a *new* parameters PyTree.

print("\n" + "="*30 + "\n")



# --- Module 4 Summary ---
# - PyTrees are nested Python containers (tuples, lists, dicts) with arrays (or other PyTrees) as leaves.
#   They are JAX's standard way to handle structured data like model parameters.
# - JAX transformations (`jit`, `grad`, `vmap`, `value_and_grad`) operate seamlessly on PyTrees,
#   applying their logic to the leaves while preserving the container structure.
# - `jax.grad` applied to a function with PyTree arguments returns gradients as a PyTree with the *identical structure*.
# - `jax.vmap` uses `in_axes` with `None` to indicate arguments (like parameter PyTrees) that should be broadcast/reused across the mapped dimension, not mapped over.
# - Composing these transformations (`jit(value_and_grad(vmap(...)))`) allows building efficient,
#   batched, differentiable functions that work on complex parameter structures from simple components.
# - This contrasts with PyTorch's `nn.Module` structure and stateful parameter updates. JAX encourages
#   handling parameters and gradients as explicit, immutable PyTree values passed through functions.

# End of Module 4