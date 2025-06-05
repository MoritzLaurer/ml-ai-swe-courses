# 06_parallelization_pmap.py

# Welcome to Module 6 of the JAX Learning Curriculum!
# Objective: Learn the basics of data parallelism using `jax.pmap` to distribute
#            computation across multiple devices (GPUs/TPUs).
# Theme Integration: We'll parallelize our MLP training step across available devices.

import jax
import jax.numpy as jnp
import jax.random as random
import optax
import time
import os # To potentially suppress GPU warnings if needed

# Optional: Suppress TensorFlow warnings if JAX uses TF for GPU management
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Optional: Specify CPU if no GPU/TPU or if you want to force CPU
# jax.config.update('jax_platform_name', 'cpu')



# --- 6.1 Device Discovery ---
# Let's see how many devices JAX detects.

print("--- 6.1 Device Discovery ---")
num_devices = jax.local_device_count()
devices = jax.devices()
print(f"Number of local JAX devices detected: {num_devices}")
print(f"Detected devices: {devices}")

if num_devices == 1:
    print("\nNOTE: Running on a single device. `pmap` will execute,")
    print("      but without actual parallelism speedup.")
    print("      The code structure learned here is still applicable to multi-device setups.")

print("\n" + "="*30 + "\n")



# --- 6.2 Recap: Setup from Module 5 ---
# We need the initialization, loss function, and optimizer setup.

# Model dimensions
input_dim = 4
hidden_dim = 8
output_dim = 2

# Initialization function (same as before)
def init_mlp_params(key, input_d, hidden_d, output_d):
  key, key_l1_w, key_l1_b, key_l2_w, key_l2_b = random.split(key, 5)
  scale1 = jnp.sqrt(2.0 / input_d); scale2 = jnp.sqrt(2.0 / hidden_d)
  params = {
      'linear1': {'W': random.normal(key_l1_w, (input_d, hidden_d))*scale1, 'b': random.normal(key_l1_b, (hidden_d,))*scale1},
      'linear2': {'W': random.normal(key_l2_w, (hidden_d, output_d))*scale2, 'b': random.normal(key_l2_b, (output_d,))*scale2}
  }
  return params

# Forward pass and loss (using vmapped version for batch avg)
def single_forward_mlp(params, x_single):
    l1_out = jnp.matmul(x_single, params['linear1']['W']) + params['linear1']['b']; l1_act = jax.nn.relu(l1_out)
    return jnp.matmul(l1_act, params['linear2']['W']) + params['linear2']['b']
def single_mse_loss_mlp(params, x_single, y_target_single):
    y_pred_single = single_forward_mlp(params, x_single); return jnp.mean(jnp.square(y_pred_single - y_target_single))
def average_batch_loss_mlp_vmapped(params, x_batch, y_target_batch):
    per_example_losses = jax.vmap(single_mse_loss_mlp, in_axes=(None, 0, 0))(params, x_batch, y_target_batch); return jnp.mean(per_example_losses)

# Optimizer setup
learning_rate = 1e-3
optimizer = optax.adam(learning_rate=learning_rate)

# Initial parameters and optimizer state (will be replicated across devices)
key = random.PRNGKey(42)
key, init_key = random.split(key)
mlp_params = init_mlp_params(init_key, input_dim, hidden_dim, output_dim)
opt_state = optimizer.init(mlp_params)

print("--- 6.2 Recap: Initial Params and Opt State ---")
print(f"Parameter shapes:\n{jax.tree.map(lambda p: p.shape, mlp_params)}")
print(f"Initial Opt State:\n{jax.tree.map(lambda x: x.shape if hasattr(x, 'shape') else x, opt_state)}")

print("\n" + "="*30 + "\n")



# --- 6.3 Preparing Data for `pmap` ---
# `pmap` expects inputs to have an additional leading dimension equal to the
# number of devices. Each slice along this dimension goes to one device.
# We need to reshape our global batch.

global_batch_size = 16 * num_devices # Make global batch size divisible by num_devices
local_batch_size = global_batch_size // num_devices

key, x_key, y_key = random.split(key, 3)
global_batch_x = random.normal(x_key, (global_batch_size, input_dim))
global_batch_y = random.normal(y_key, (global_batch_size, output_dim))

print("--- 6.3 Preparing Data for pmap ---")
print(f"Global batch size: {global_batch_size}")
print(f"Local batch size per device: {local_batch_size}")
print(f"Global batch X shape: {global_batch_x.shape}")

def shard_batch(batch):
  """Reshapes the batch to have a leading device dimension."""
  return batch.reshape((num_devices, local_batch_size) + batch.shape[1:])

sharded_batch_x = shard_batch(global_batch_x)
sharded_batch_y = shard_batch(global_batch_y)

print(f"\nSharded batch X shape: {sharded_batch_x.shape}") # Should be (num_devices, local_batch_size, input_dim)
print(f"Sharded batch Y shape: {sharded_batch_y.shape}") # Should be (num_devices, local_batch_size, output_dim)

print("\n" + "="*30 + "\n")



# --- 6.4 Modifying the Training Step for `pmap` Collectives ---
# The function to be `pmap`-ed needs to handle cross-device communication
# explicitly using collective operations like `jax.lax.pmean`.

# We need the non-jitted value/grad function to modify it
_value_and_grad_fn = jax.value_and_grad(average_batch_loss_mlp_vmapped, argnums=0)

def pmap_training_step(params, opt_state, batch_x, batch_y):
  """
  Performs one training step, designed to be pmapped.
  Includes gradient averaging using pmean.
  """
  # Calculate loss and gradients *locally* on each device
  local_loss, local_grads = _value_and_grad_fn(params, batch_x, batch_y)

  # !!! Average gradients across all devices !!!
  # `axis_name` must match the name given to `pmap`. 'batch' is common.
  # `pmean` computes the mean over the devices mapped by pmap.
  grads = jax.lax.pmean(local_grads, axis_name='batch')

  # Optional: Average the loss too for consistent reporting across devices
  loss = jax.lax.pmean(local_loss, axis_name='batch')

  # Optimizer update step (uses the *averaged* gradients)
  updates, new_opt_state = optimizer.update(grads, opt_state, params)
  new_params = optax.apply_updates(params, updates)

  return new_params, new_opt_state, loss

print("--- 6.4 Modified training_step with jax.lax.pmean ---")
print("Gradients and loss will be averaged across devices inside the step.")

print("\n" + "="*30 + "\n")



# --- 6.5 Applying `jax.pmap` ---
# `pmap` takes the function and an `axis_name`.

# Static arguments: Tell pmap that the optimizer definition is static and doesn't change.
# This can sometimes help compilation, though not strictly necessary here.
# We could also define `optimizer` inside `pmap_training_step`.
# pmapped_training_step = jax.pmap(pmap_training_step, axis_name='batch', static_broadcasted_argnums=(...))

pmapped_training_step = jax.pmap(
    pmap_training_step,
    axis_name='batch', # This name MUST match the one used in jax.lax.pmean
    in_axes=(None, None, 0, 0) # Specify replication (None) or mapping (0) for each arg
)

print("--- 6.5 Applying jax.pmap ---")
print(f"`pmapped_training_step` created by applying `jax.pmap`.")

# --- First call and Replication ---
# The first time `pmapped_training_step` is called:
# 1. The function is compiled for the devices.
# 2. Non-sharded inputs (like `mlp_params`, `opt_state`) are automatically
#    replicated (copied) to each device.

print("\nExecuting first pmapped step (includes compile time and replication)...")
start_time = time.time()
# Inputs: params/opt_state are NOT sharded, batch_x/y ARE sharded
pmapped_params, pmapped_opt_state, pmapped_loss = pmapped_training_step(
    mlp_params, opt_state, sharded_batch_x, sharded_batch_y
)
end_time = time.time()
print(f"First step execution time: {end_time - start_time:.4f}s")

# --- Understanding Outputs ---
# The outputs of a pmapped function are also replicated/sharded across devices.
# They have a leading dimension equal to `num_devices`.
print("\nOutput shapes after pmap:")
print(f"Pmapped Params shapes:\n{jax.tree.map(lambda p: p.shape, pmapped_params)}") # Each leaf shape starts with `num_devices`
print(f"Pmapped Opt State shapes:\n{jax.tree.map(lambda x: x.shape if hasattr(x, 'shape') else x, pmapped_opt_state)}")
print(f"Pmapped Loss shape: {pmapped_loss.shape}") # Shape is (num_devices,)

# All values along the device axis should be identical because parameters were
# replicated and gradients were averaged.
print(f"\nLoss value on each device: {pmapped_loss}")
# Check one parameter leaf across devices:
print(f"W1[0, 0] on each device: {pmapped_params['linear1']['W'][:, 0, 0]}") # Access device dim with [:, ...]

# For subsequent steps or saving, usually only need the values from one device (e.g., index 0)
params_for_next_step = jax.tree.map(lambda x: x[0], pmapped_params)
opt_state_for_next_step = jax.tree.map(lambda x: x[0], pmapped_opt_state)
loss_reported = pmapped_loss[0]
print(f"\nLoss value reported (from device 0): {loss_reported:.4f}")

print("\n" + "="*30 + "\n")



# --- 6.6 Simulating Training Loop with `pmap` ---
# The loop looks similar, but manages sharded data and extracts results.

num_steps = 100
# Use initial params/state for the loop start
current_params = mlp_params
current_opt_state = opt_state

print(f"--- 6.6 Simulating pmapped Training Loop ({num_steps} steps) ---")
start_loop_time = time.time()

for step in range(num_steps):
  # In a real loop: get global batch, shard it
  key, x_key, y_key = random.split(key, 3)
  global_batch_x = random.normal(x_key, (global_batch_size, input_dim))
  global_batch_y = random.normal(y_key, (global_batch_size, output_dim))
  sharded_batch_x = shard_batch(global_batch_x)
  sharded_batch_y = shard_batch(global_batch_y)

  # Execute the pmapped step (params/state are replicated automatically)
  pmapped_params, pmapped_opt_state, pmapped_loss = pmapped_training_step(
      current_params, current_opt_state, sharded_batch_x, sharded_batch_y
  )

  # *** Crucial: Update loop state with values from one device ***
  # Otherwise JAX tries to feed replicated values back in, causing shape errors or recompiles.
  current_params = jax.tree.map(lambda x: x[0], pmapped_params)
  current_opt_state = jax.tree.map(lambda x: x[0], pmapped_opt_state)
  loss = pmapped_loss[0] # Get loss from one device

  if (step + 1) % 10 == 0:
    # This print happens on the host, after the device computation is done
    print(f"Step {step+1}/{num_steps}, Loss: {loss:.4f}")

end_loop_time = time.time()
print(f"\nFinished {num_steps} pmapped steps in {end_loop_time - start_loop_time:.4f}s")
print(f"Final Loss: {loss:.4f}")


# ** PyTorch Contrast **
# - `torch.nn.DataParallel` (DP): Wraps the model, simpler API but often slower due to Python overhead/GIL. Handles data splitting and gradient reduction somewhat automatically.
# - `torch.nn.DistributedDataParallel` (DDP): Standard approach. Requires setting up process groups (`torch.distributed.init_process_group`). Each process runs the script. DDP handles gradient sync more efficiently. Setup is more involved than `pmap`.
# - Communication: DDP often implicitly syncs gradients after `.backward()`. `pmap` uses explicit collectives like `jax.lax.pmean` *within* the function definition, tied to an `axis_name`.
# - Integration: `pmap` is a core JAX transformation that composes with `jit`. PyTorch DDP interacts with the autograd engine.
# - Flexibility: `pmap` is the basis for SPMD programming. JAX offers more advanced sharding mechanisms (`shard_map`, `mesh_utils`, partitioning APIs) for complex parallelism (model/pipeline) beyond simple data parallelism, often used in large model training libraries.

print("\n" + "="*30 + "\n")

# --- Module 6 Summary ---
# - `jax.pmap` is a JAX transformation for SPMD-style data parallelism across multiple devices.
# - It compiles a function to run on all devices, expecting inputs to have a leading device dimension.
# - Data must be manually *sharded* (split) along the device dimension before being passed to the `pmap`-ed function (e.g., using reshape).
# - Non-sharded arguments (like parameters, optimizer state in basic data parallel) are automatically *replicated* to all devices by `pmap`.
# - Cross-device communication (like gradient averaging) must be done explicitly *inside* the `pmap`-ed function using JAX collectives (e.g., `jax.lax.pmean(value, axis_name='...')`).
# - The `axis_name` links the `pmap` transformation to the collectives used within it.
# - Outputs from `pmap` also have a leading device dimension. Usually, you extract the result from one device (e.g., index 0) for use in the Python host loop or for saving.
# - Even on a single device, `pmap` runs, allowing you to develop and test the parallel code structure.

# End of Module 6