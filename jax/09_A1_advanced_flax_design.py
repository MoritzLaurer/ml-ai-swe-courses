# A1_advanced_flax_design.py

# Welcome to Advanced Module A.1: Advanced Flax Module Design!
# Objective: Learn to handle stateful layers (BatchNorm), manage variable
#            collections, and build custom Flax modules.
# Theme Integration: We'll modify our MLP to include BatchNorm and create
#                    a custom residual block.

import jax
import jax.numpy as jnp
import jax.random as random
import flax.linen as nn
import optax # For optimizer context
import time
from typing import Sequence, Dict, Any # For type hinting



# --- A1.1 Stateful Layers: Batch Normalization ---
print("--- A1.1 Stateful Layers: BatchNorm ---")
# Some layers need to maintain internal state beyond trainable parameters.
# BatchNorm is a prime example: it calculates running means and variances
# during training, which are used during evaluation.
# Flax handles this via multiple "Variable Collections".
# - 'params': Default collection for trainable parameters.
# - 'batch_stats': Default collection for BatchNorm's running averages.
# - Other collections can be defined for different kinds of state.

# --- How Flax Manages State Updates ---
# 1. Initialization (`init`): Returns all necessary collections.
#    `variables = model.init(...)` -> `variables` is often `{'params': ..., 'batch_stats': ...}`
# 2. Application (`apply`): Can update state if requested via `mutable`.
#    - Inference (`mutable=False`): Uses existing state (e.g., running averages
#      for BatchNorm if `use_running_average=True`). Returns only model output.
#    - Training (`mutable=['batch_stats']` or `mutable=True`): Allows updates to
#      the specified collections. Returns `(output, updated_variables)`.

# --- Example: MLP with BatchNorm ---
class FlaxMLPWithBatchNorm(nn.Module):
  hidden_dims: Sequence[int] # Can have multiple hidden layers
  output_dim: int
  train: bool # Flag to control BatchNorm/Dropout behavior

  @nn.compact
  def __call__(self, x):
    for i, h_dim in enumerate(self.hidden_dims):
      x = nn.Dense(features=h_dim, name=f'dense_{i}')(x)
      # Apply BatchNorm AFTER Dense, BEFORE activation
      # `use_running_average = not self.train` tells BN whether to use
      # batch statistics (train=True) or running averages (train=False).
      x = nn.BatchNorm(use_running_average=not self.train,
                         name=f'bn_{i}')(x)
      x = nn.relu(x)
    # Final output layer
    x = nn.Dense(features=self.output_dim, name='output_dense')(x)
    return x

# --- Initialization ---
print("Initializing MLP with BatchNorm...")
key = random.PRNGKey(101)
key, init_key = random.split(key)

input_dim = 4
hidden_dims = [8, 6] # Two hidden layers
output_dim = 2
dummy_input = jnp.ones((1, input_dim))

# Instantiate for training mode initially to see structure
mlp_bn_train = FlaxMLPWithBatchNorm(hidden_dims=hidden_dims, output_dim=output_dim, train=True)
variables = mlp_bn_train.init(init_key, dummy_input)

params = variables['params']
batch_stats = variables['batch_stats']

print("\nInitialized Variable Collections:")
print("  'params' keys/shapes:", jax.tree_map(lambda p: p.shape, params))
print("  'batch_stats' keys/shapes:", jax.tree_map(lambda s: s.shape, batch_stats)) # mean, var per BN layer

# --- Training `apply` call ---
# During training, we need to update `batch_stats`.
print("\nSimulating Training `apply` call...")
# Note: Pass `train=True` to the *instance* if using the flag approach,
# or pass it to `apply` if `train` is an argument to `__call__`.
mlp_bn_train_instance = FlaxMLPWithBatchNorm(hidden_dims=hidden_dims, output_dim=output_dim, train=True)

# Combine variables needed for apply
apply_vars_train = {'params': params, 'batch_stats': batch_stats}

# Use `mutable=['batch_stats']` to allow updates to this collection
output_train, updated_state = mlp_bn_train_instance.apply(
    apply_vars_train,
    dummy_input,
    mutable=['batch_stats'] # Specify which collections can be mutated
)
new_batch_stats = updated_state['batch_stats']

print(f"  Output shape (train): {output_train.shape}")
print(f"  Returned updated 'batch_stats': {True if 'batch_stats' in updated_state else False}")
# Compare original and new batch stats (they should differ slightly after one step)
print(f"  Original BN mean (layer 0): {batch_stats['bn_0']['mean'][0]:.4f}")
print(f"  Updated BN mean (layer 0):  {new_batch_stats['bn_0']['mean'][0]:.4f}")

# --- Evaluation `apply` call ---
# During evaluation, use running averages and DON'T update batch_stats.
print("\nSimulating Evaluation `apply` call...")
mlp_bn_eval_instance = FlaxMLPWithBatchNorm(hidden_dims=hidden_dims, output_dim=output_dim, train=False)

# Combine variables needed for apply (use the *updated* stats from training)
apply_vars_eval = {'params': params, 'batch_stats': new_batch_stats}

# `mutable=False` is default, no need to specify. Module runs immutably.
# The `train=False` flag passed to the instance ensures BN uses running averages.
output_eval = mlp_bn_eval_instance.apply(apply_vars_eval, dummy_input)

print(f"  Output shape (eval): {output_eval.shape}")
print(f"  Output value (eval, different from train): {output_eval}")

# --- Integration into Training Step (Conceptual) ---
# The `training_step` function would now need to:
# - Accept `params` AND `batch_stats` as input state.
# - Call `model.apply(..., mutable=['batch_stats'])` inside `value_and_grad`.
#   *Note:* `value_and_grad` needs careful handling here. Often, the loss function
#    passed to `value_and_grad` computes *only* the loss, and the state update is
#    done separately or using `jax.value_and_grad(..., has_aux=True)` if the loss
#    function also returns the new state.
# - Extract `new_batch_stats` from the `updated_state`.
# - Return `new_params`, `new_opt_state`, AND `new_batch_stats`.
# The outer loop manages all three state PyTrees.
# The `eval_step` would pass `train=False` (or `use_running_average=True`)
# and use `mutable=False`.

# ** PyTorch Contrast **
# - PyTorch handles BatchNorm state implicitly via `model.train()` and `model.eval()` modes.
# - `model.eval()` switches BatchNorm (& Dropout) to use running averages and disables gradient calculation setup in some contexts.
# - Flax requires explicit state management: passing `'batch_stats'` in/out and using `mutable`.

print("\n" + "="*30 + "\n")



# --- A1.2 Custom Flax Modules ---
print("--- A1.2 Custom Flax Modules ---")
# You can easily create your own reusable modules by inheriting from `nn.Module`.
# This helps organize complex architectures.

# Example: A Simple Residual Block
class ResidualBlock(nn.Module):
  features: int # Number of features in the dense layers
  train: bool   # Pass training flag for BatchNorm

  @nn.compact
  def __call__(self, x):
    # Store residual connection
    residual = x

    # Main path
    y = nn.Dense(features=self.features, name='dense_1')(x)
    y = nn.BatchNorm(use_running_average=not self.train, name='bn_1')(y)
    y = nn.relu(y)
    y = nn.Dense(features=x.shape[-1], name='dense_2')(y) # Project back to input dim
    y = nn.BatchNorm(use_running_average=not self.train, name='bn_2')(y)

    # Add residual connection
    output = y + residual
    # Optional: Apply final activation AFTER residual connection
    # output = nn.relu(output)
    return output

# --- Using the Custom Module ---
class ResNetMLP(nn.Module):
  num_blocks: int
  hidden_features: int
  output_dim: int
  train: bool

  @nn.compact
  def __call__(self, x):
    # Initial projection if needed (e.g., if input_dim != hidden_features)
    x = nn.Dense(features=self.hidden_features, name='input_proj')(x)
    x = nn.relu(x)

    # Apply residual blocks
    for i in range(self.num_blocks):
      x = ResidualBlock(features=self.hidden_features, train=self.train, name=f'resblock_{i}')(x)
      # Maybe add activation after each block
      x = nn.relu(x)

    # Final output layer
    x = nn.Dense(features=self.output_dim, name='output_proj')(x)
    return x

# --- Initialization & Application ---
print("Initializing ResNetMLP (contains ResidualBlock)...")
key, res_init_key = random.split(key)
dummy_res_input = jnp.ones((1, input_dim)) # Same dummy input

resnet_mlp_train = ResNetMLP(num_blocks=2, hidden_features=8, output_dim=output_dim, train=True)
res_variables = resnet_mlp_train.init(res_init_key, dummy_res_input)

res_params = res_variables['params']
res_batch_stats = res_variables['batch_stats']

print("\nResNetMLP Initialized Variable Collections:")
print("  'params' structure:")
# Use `repr` for a more detailed view of the nested structure
print(repr(jax.tree_map(lambda p: p.shape, res_params)))
print("\n  'batch_stats' structure:")
print(repr(jax.tree_map(lambda s: s.shape, res_batch_stats)))

print("\nApplying ResNetMLP...")
res_apply_vars = {'params': res_params, 'batch_stats': res_batch_stats}
res_output_train, res_updated_state = resnet_mlp_train.apply(
    res_apply_vars, dummy_res_input, mutable=['batch_stats']
)
print(f"  Output shape (train): {res_output_train.shape}")


print("\n" + "="*30 + "\n")



# --- A1.3 Parameter Initialization (Briefly) ---
print("--- A1.3 Parameter Initialization ---")
# Flax layers accept initializer arguments (e.g., `kernel_init`, `bias_init`).
# Common initializers are in `flax.linen.initializers` (often aliased as `nn.initializers`).

class InitializedDense(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        # Example: Use Xavier uniform for kernel, zeros for bias
        kernel_init_fn = nn.initializers.xavier_uniform()
        # Alternative: nn.initializers.lecun_normal(), nn.initializers.kaiming_uniform(), etc.
        bias_init_fn = nn.initializers.zeros

        dense_layer = nn.Dense(features=self.features,
                               kernel_init=kernel_init_fn,
                               bias_init=bias_init_fn)
        return dense_layer(x)

# Initialization would use these functions internally
key, init_dense_key = random.split(key)
init_dense_module = InitializedDense(features=5)
init_dense_params = init_dense_module.init(init_dense_key, jnp.ones((1, 3)))['params']

print("Example Dense layer parameters with specific initializers:")
print(jax.tree_map(lambda p: (p.shape, p.dtype), init_dense_params))

print("\n" + "="*30 + "\n")



# --- Module A1 Summary ---
# - Stateful layers like `nn.BatchNorm` require managing state beyond trainable parameters.
# - Flax uses Variable Collections (e.g., `'params'`, `'batch_stats'`) to store different kinds of state.
# - `model.init()` returns all necessary collections.
# - `model.apply()` uses the `mutable` argument during training to allow updates to specified collections (e.g., `mutable=['batch_stats']`). It returns `(output, updated_variables)`.
# - During evaluation (`mutable=False`), `apply` uses the provided state immutably (e.g., BatchNorm uses running averages if `use_running_average=True`).
# - Training steps must manage the input and output state PyTrees (e.g., `params`, `batch_stats`, `opt_state`).
# - Custom, reusable modules can be created by inheriting from `flax.linen.Module`, defining submodules in `setup` or `@nn.compact`, and logic in `__call__`.
# - Parameter initializers can be specified using functions from `flax.linen.initializers`.
# - This explicit state handling is core to Flax and contrasts with PyTorch's more implicit state updates tied to `model.train()/eval()`.

# End of Module A1