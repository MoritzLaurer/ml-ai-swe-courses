# 05_optimization_optax.py

# Welcome to Module 5 of the JAX Learning Curriculum!
# Objective: Learn how to update model parameters using gradients with the Optax
#            library, managing optimizer state explicitly in JAX's functional style.
# Theme Integration: We'll use the gradients computed for our 2-layer MLP and
#                    apply updates using the Adam optimizer from Optax.

import jax
import jax.numpy as jnp
import jax.random as random
import optax # The standard JAX optimization library
import time



# --- 5.1 Recap: MLP Setup and Gradient Function ---
# Reuse the MLP definition, initialization, and the jitted value/gradient function
# from Module 4.

# Model dimensions
input_dim = 4
hidden_dim = 8
output_dim = 2

# Initialization function
def init_mlp_params(key, input_d, hidden_d, output_d):
  key, key_l1_w, key_l1_b, key_l2_w, key_l2_b = random.split(key, 5)
  scale1 = jnp.sqrt(2.0 / input_d); scale2 = jnp.sqrt(2.0 / hidden_d)
  params = {
      'linear1': {'W': random.normal(key_l1_w, (input_d, hidden_d))*scale1, 'b': random.normal(key_l1_b, (hidden_d,))*scale1},
      'linear2': {'W': random.normal(key_l2_w, (hidden_d, output_d))*scale2, 'b': random.normal(key_l2_b, (output_d,))*scale2}
  }
  return params

# Forward pass and loss functions (single-example versions for vmap)
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

# Average batch loss using vmap
def average_batch_loss_mlp_vmapped(params, x_batch, y_target_batch):
    per_example_losses = jax.vmap(single_mse_loss_mlp, in_axes=(None, 0, 0))(params, x_batch, y_target_batch)
    return jnp.mean(per_example_losses)

# Get the jitted value-and-gradient function w.r.t. parameters (arg 0)
value_and_grad_fn = jax.value_and_grad(average_batch_loss_mlp_vmapped, argnums=0)
jitted_value_and_grad = jax.jit(value_and_grad_fn)

# Initialize parameters
key = random.PRNGKey(42)
key, init_key = random.split(key)
mlp_params = init_mlp_params(init_key, input_dim, hidden_dim, output_dim)

# Dummy batch data
batch_size = 16
key, x_key, y_key = random.split(key, 3)
batch_x = random.normal(x_key, (batch_size, input_dim))
batch_y = random.normal(y_key, (batch_size, output_dim))

# Calculate initial loss and gradients
initial_loss, initial_grads = jitted_value_and_grad(mlp_params, batch_x, batch_y)

print("--- 5.1 Recap: Initial State ---")
print(f"Parameter PyTree shapes:\n{jax.tree.map(lambda p: p.shape, mlp_params)}")
print(f"Initial batch Loss: {initial_loss:.4f}")
print(f"Initial gradient PyTree shapes:\n{jax.tree.map(lambda g: g.shape, initial_grads)}")

print("\n" + "="*30 + "\n")



# --- 5.2 Introduction to Optax ---
# Optax provides common optimizers (Adam, SGD, RMSprop, etc.) and building blocks.
# Key idea: Optimizers are often stateful (e.g., Adam tracks momentum).
# In JAX, this state must be handled explicitly.

print("--- 5.2 Setting up the Optax Optimizer ---")

# 1. Define the optimizer
learning_rate = 1e-3
# optimizer = optax.sgd(learning_rate=learning_rate)
optimizer = optax.adam(learning_rate=learning_rate) # Adam is often a good default
print(f"Optimizer defined: {optimizer}")

# 2. Initialize the optimizer state
# The state is also a PyTree, often mirroring the parameters structure,
# to hold things like momentum vectors for each parameter.
opt_state = optimizer.init(mlp_params)
print("\nInitial Optimizer State PyTree structure:")
print(jax.tree.map(lambda x: x.shape if hasattr(x, 'shape') else x, opt_state))
# Note: For Adam, the state includes 'mu' (momentum) and 'nu' (variance tracking),
# matching the parameter structure, plus a step count.

print("\n" + "="*30 + "\n")



# --- 5.3 The Optimization Step ---
# Optax separates calculating updates from applying them.

# 1. Calculate Updates: `optimizer.update`
# Takes (gradients, optimizer_state, optional_params) -> (updates, new_optimizer_state)
# `updates` is a PyTree of the *changes* to be applied (e.g., -lr * scaled_gradient).
updates, new_opt_state = optimizer.update(initial_grads, opt_state, mlp_params)

print("--- 5.3 Calculating and Applying Updates ---")
print("Calculated Updates PyTree structure (matches grads/params):")
print(jax.tree.map(lambda u: u.shape, updates))
print("\nNew Optimizer State structure (same as before, values updated):")
print(jax.tree.map(lambda x: x.shape if hasattr(x, 'shape') else x, new_opt_state))

# 2. Apply Updates: `optax.apply_updates`
# Takes (params, updates) -> new_params
# Typically computes `params + updates` element-wise for each leaf.
new_mlp_params = optax.apply_updates(mlp_params, updates)

print("\nNew Parameters PyTree structure (same as before, values updated):")
print(jax.tree.map(lambda p: p.shape, new_mlp_params))

# Verify parameters have changed (compare one leaf)
print(f"\nOriginal W1[0, 0]: {mlp_params['linear1']['W'][0, 0]:.6f}")
print(f"Updated W1[0, 0]:  {new_mlp_params['linear1']['W'][0, 0]:.6f}")

# And the optimizer state has changed (compare count)
print(f"\nOriginal opt_state count: {opt_state.count}") # AdamState has a count attribute
print(f"New opt_state count:      {new_opt_state.count}")

print("\n" + "="*30 + "\n")



# --- 5.4 Creating the `training_step` Function ---
# Let's bundle the gradient calculation and optimizer update into a single,
# pure function that can be JIT-compiled.

# Use the *non-jitted* value_and_grad function inside, as we'll jit the whole step.
_value_and_grad_fn = jax.value_and_grad(average_batch_loss_mlp_vmapped, argnums=0)

def training_step(params, opt_state, batch_x, batch_y):
  """Performs one step of training."""
  # 1. Calculate loss and gradients
  loss, grads = _value_and_grad_fn(params, batch_x, batch_y)
  # 2. Calculate updates and new optimizer state
  updates, new_opt_state = optimizer.update(grads, opt_state, params)
  # 3. Apply updates to parameters
  new_params = optax.apply_updates(params, updates)
  # Return updated state and metrics
  return new_params, new_opt_state, loss

# JIT-compile the entire training step
jitted_training_step = jax.jit(training_step)

print("--- 5.4 JIT-Compiled Training Step ---")
print("`jitted_training_step` function created.")

# Test one step
start_time = time.time()
(params_after_step1,
 opt_state_after_step1,
 loss_after_step1) = jitted_training_step(mlp_params, opt_state, batch_x, batch_y)
end_time = time.time()

print(f"\nExecuted one jitted step in {end_time-start_time:.6f}s")
print(f"Loss after 1 step: {loss_after_step1:.4f}")
print(f"Optimizer state count after 1 step: {opt_state_after_step1.count}")
print(f"W1[0, 0] after 1 step: {params_after_step1['linear1']['W'][0, 0]:.6f}")

print("\n" + "="*30 + "\n")



# --- 5.5 Simulating a Training Loop ---
# The actual training loop happens outside the jitted function in standard Python.
# It manages the state (params, opt_state) across iterations.

num_steps = 100
# Use the parameters and opt_state from *after* the first step calculation above
current_params = params_after_step1
current_opt_state = opt_state_after_step1

print(f"--- 5.5 Simulating Training Loop ({num_steps} steps) ---")
start_loop_time = time.time()

for step in range(num_steps):
  # In a real scenario, get a new batch of data here
  # key, x_key, y_key = random.split(key, 3)
  # batch_x = random.normal(x_key, (batch_size, input_dim))
  # batch_y = random.normal(y_key, (batch_size, output_dim))
  # We reuse the same batch for simplicity here

  # Execute the jitted training step
  current_params, current_opt_state, loss = jitted_training_step(
      current_params, current_opt_state, batch_x, batch_y
  )

  # Log progress (e.g., every 10 steps)
  if (step + 1) % 10 == 0:
    print(f"Step {step+1}/{num_steps}, Loss: {loss:.4f}")

end_loop_time = time.time()
print(f"\nFinished {num_steps} steps in {end_loop_time - start_loop_time:.4f}s")
print(f"Final Loss: {loss:.4f}")
print(f"Final W1[0, 0]: {current_params['linear1']['W'][0, 0]:.6f}")
print(f"Final optimizer state count: {current_opt_state.count}")


# ** PyTorch Contrast **
# - Initialization: Similar (`model = Model()`, `optimizer = optim.Adam(model.parameters())`).
# - Training Loop:
#   - `optimizer.zero_grad()` # Clear previous gradients stored in tensor attributes
#   - `outputs = model(inputs)`
#   - `loss = criterion(outputs, targets)`
#   - `loss.backward()` # Compute gradients and store in `.grad` attributes
#   - `optimizer.step()` # Update model parameters in-place using `.grad` and internal state
# - State Management: PyTorch implicitly manages gradients (`.grad`) and optimizer state
#   within the model and optimizer objects. Updates happen in-place.
# - JAX/Optax: State (params, opt_state) is explicit. The `training_step` is pure;
#   it takes state in and returns *new* state out. The Python loop manages the state flow.
#   No in-place updates, no implicit gradient accumulation.

print("\n" + "="*30 + "\n")



# --- Module 5 Summary ---
# - Optax is the standard library for optimization in JAX.
# - Optimizers like Adam are stateful; Optax requires explicit handling of this state.
# - Workflow:
#   1. Define optimizer: `optimizer = optax.adam(...)`
#   2. Initialize state: `opt_state = optimizer.init(params)` (params is a PyTree)
#   3. In the training step (often jitted):
#      a. Compute `(loss, grads) = value_and_grad_fn(params, ...)`
#      b. Compute `(updates, new_opt_state) = optimizer.update(grads, opt_state, params)`
#      c. Compute `new_params = optax.apply_updates(params, updates)`
#      d. Return `(new_params, new_opt_state, loss)`
# - The training loop in Python passes the state (`params`, `opt_state`) into the pure,
#   jitted `training_step` function and receives the updated state back for the next iteration.
# - This explicit state management contrasts with PyTorch's implicit state handling and in-place updates.

# End of Module 5