# 08_finetuning_loop.py

# Welcome to Module 8 of the JAX Learning Curriculum!
# Objective: Combine all learned concepts (Flax models, HF integration, Optax, pmap)
#            to structure a basic fine-tuning loop for a transformer model.
# Theme Integration: Define a sequence classification model using DistilBERT + a
#                    classification head and sketch the training loop structure.

import jax
import jax.numpy as jnp
import jax.random as random
import flax.linen as nn
import optax
import time
from functools import partial # For defining train_step with static args

# Ensure transformers, flax, optax are installed
from transformers import AutoTokenizer, FlaxAutoModel



# --- 8.1 Setup: Model, Tokenizer, Optimizer ---

print("--- 8.1 Setup ---")

# --- Device Setup ---
num_devices = jax.local_device_count()
devices = jax.devices()
print(f"Using {num_devices} devices: {devices}")

# --- Model Configuration ---
model_name = "distilbert-base-uncased"
num_classes = 3 # Example: 3 classes for classification

# --- Load Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Tokenizer '{model_name}' loaded.")

# --- Define the Fine-tuning Model (Transformer + Classification Head) ---
class SequenceClassifier(nn.Module):
  model_name: str
  num_classes: int

  def setup(self):
    # Load the pre-trained transformer body (Flax version)
    self.transformer = FlaxAutoModel.from_pretrained(self.model_name)
    # Define the classification head
    self.classifier = nn.Dense(features=self.num_classes)

  def __call__(self, input_ids, attention_mask, train: bool = True):
    # Pass inputs through the transformer body
    # We access params externally via `apply`, but can use `self.transformer` here
    transformer_outputs = self.transformer(
        input_ids=input_ids, attention_mask=attention_mask
    )
    # Use the [CLS] token embedding (first token) for classification
    cls_embedding = transformer_outputs.last_hidden_state[:, 0]

    # TODO: Add dropout here during training if needed:
    # if train:
    #   cls_embedding = nn.Dropout(rate=0.1)(cls_embedding, deterministic=not train)

    # Pass the [CLS] embedding through the classifier head
    logits = self.classifier(cls_embedding)
    return logits

# Instantiate the combined model
classifier_model = SequenceClassifier(model_name=model_name, num_classes=num_classes)
print("SequenceClassifier model defined.")

# --- Initialize Parameters ---
# Use a dummy input shape that matches typical tokenized output
max_len = 64 # Example max sequence length
dummy_input_ids = jnp.ones((1, max_len), dtype=jnp.int32)
dummy_attention_mask = jnp.ones((1, max_len), dtype=jnp.int32)

key = random.PRNGKey(42)
key, init_key = random.split(key)

# Initialize *all* parameters (transformer + classifier head)
# The transformer params will load pre-trained values from `setup`
# The classifier head params will be randomly initialized by `nn.Dense.init`
variables = classifier_model.init(init_key, dummy_input_ids, dummy_attention_mask)
# `init` returns a dict potentially including 'params', 'batch_stats', etc.
# We primarily care about 'params' for basic fine-tuning.
params = variables['params']
print("Model parameters initialized.")
print("Parameter PyTree structure (top levels):")
print(jax.tree.map(lambda x: x.shape if hasattr(x, 'shape') else x, params))

# --- Optimizer Setup ---
learning_rate = 5e-5 # Common learning rate for fine-tuning transformers
optimizer = optax.adamw(learning_rate=learning_rate) # AdamW is often used
opt_state = optimizer.init(params)
print("Optimizer (AdamW) initialized.")

print("\n" + "="*30 + "\n")



# --- 8.2 Loss Function ---
print("--- 8.2 Loss Function ---")

def cross_entropy_loss(logits, labels):
  # Simple cross-entropy loss for classification
  one_hot_labels = jax.nn.one_hot(labels, num_classes=num_classes)
  return -jnp.sum(one_hot_labels * jax.nn.log_softmax(logits), axis=-1)

def compute_loss(params, batch_input_ids, batch_attention_mask, batch_labels):
  """Computes average loss for a batch."""
  # Get model predictions (logits)
  logits = classifier_model.apply({'params': params},
                                  input_ids=batch_input_ids,
                                  attention_mask=batch_attention_mask,
                                  train=True) # Pass train=True if model uses dropout etc.
  # Calculate loss per example
  loss = cross_entropy_loss(logits, batch_labels)
  # Return the average loss over the batch
  return jnp.mean(loss)

print("Cross-entropy loss function defined.")

print("\n" + "="*30 + "\n")



# --- 8.3 Training Step Definition ---
print("--- 8.3 Training Step Definition ---")

# Get the value and gradient function (w.r.t params)
_value_and_grad_fn = jax.value_and_grad(compute_loss, argnums=0)

def training_step(params, opt_state, batch_input_ids, batch_attention_mask, batch_labels):
  """Performs one pmapped training step with gradient averaging."""
  # Calculate local loss and gradients
  local_loss, local_grads = _value_and_grad_fn(params, batch_input_ids, batch_attention_mask, batch_labels)

  # === Gradient & Loss Averaging across devices ===
  grads = jax.lax.pmean(local_grads, axis_name='batch')
  loss = jax.lax.pmean(local_loss, axis_name='batch')
  # =============================================

  # Optimizer update
  updates, new_opt_state = optimizer.update(grads, opt_state, params)
  new_params = optax.apply_updates(params, updates)

  return new_params, new_opt_state, loss

print("`training_step` function defined (includes pmean).")

# --- Apply `pmap` ---
# Use `partial` to potentially fix static arguments like the model definition later if needed
pmapped_training_step = jax.pmap(
    training_step,
    axis_name='batch' # Name used in pmean
    # donate_argnums=(0, 1) # Optional: Allow JAX to reuse memory for params/opt_state
)
print("`pmapped_training_step` created.")

print("\n" + "="*30 + "\n")



# --- 8.4 Data Preparation (Dummy Data) ---
print("--- 8.4 Dummy Data Preparation ---")
# In a real scenario, this would involve a DataLoader loading and tokenizing data.

global_batch_size = 8 * num_devices # Ensure divisible by num_devices
local_batch_size = global_batch_size // num_devices
print(f"Global batch size: {global_batch_size}, Local batch size: {local_batch_size}")

def get_dummy_batch(key):
  """Generates a dummy batch of data."""
  key_ids, key_mask, key_lbl = random.split(key, 3)
  # Shape: (global_batch_size, max_len)
  dummy_ids = random.randint(key_ids, (global_batch_size, max_len), 0, tokenizer.vocab_size)
  dummy_mask = jnp.ones((global_batch_size, max_len), dtype=jnp.int32) # Simple mask
  # Shape: (global_batch_size,)
  dummy_lbls = random.randint(key_lbl, (global_batch_size,), 0, num_classes)
  return dummy_ids, dummy_mask, dummy_lbls

def shard_batch(batch):
  """Reshapes the batch for pmap (leading device dimension)."""
  return jax.tree.map(lambda x: x.reshape((num_devices, local_batch_size) + x.shape[1:]), batch)

# Generate one global batch and shard it
key, data_key = random.split(key)
global_ids, global_mask, global_labels = get_dummy_batch(data_key)
sharded_batch = shard_batch((global_ids, global_mask, global_labels))
print(f"Sharded batch shapes: {jax.tree.map(lambda x: x.shape, sharded_batch)}")

print("\n" + "="*30 + "\n")



# --- 8.5 Fine-Tuning Loop Structure ---
print("--- 8.5 Fine-Tuning Loop Structure ---")

num_train_steps = 50 # Short loop for demonstration

# === Replicate initial state across devices ===
# This ensures each device starts with the same parameters and optimizer state.
# While pmap can do this implicitly on the first call, explicit replication is clearer.
replicated_params = jax.device_put_replicated(params, devices)
replicated_opt_state = jax.device_put_replicated(opt_state, devices)
print("Initial parameters and optimizer state replicated across devices.")
# ============================================

start_loop_time = time.time()
# Training loop (runs on host, dispatches computation to devices via pmap)
for step in range(num_train_steps):
  # 1. Get and shard data for the step
  key, data_key = random.split(key)
  global_ids, global_mask, global_labels = get_dummy_batch(data_key)
  sharded_batch_ids, sharded_batch_mask, sharded_batch_labels = shard_batch(
      (global_ids, global_mask, global_labels)
  )

  # 2. Execute the pmapped training step
  # Pass the *replicated* state and *sharded* data
  (replicated_params,
   replicated_opt_state,
   replicated_loss) = pmapped_training_step(replicated_params,
                                            replicated_opt_state,
                                            sharded_batch_ids,
                                            sharded_batch_mask,
                                            sharded_batch_labels)
  # Note: replicated_params/opt_state contain the *updated* state, still replicated.

  # 3. Log metrics (optional)
  if (step + 1) % 10 == 0:
    # Get loss from one device (they are all the same after pmean)
    loss_value = replicated_loss[0]
    # Use jax.device_get() to bring value back to host CPU for printing/logging
    print(f"Step: {step+1}/{num_train_steps}, Loss: {jax.device_get(loss_value):.4f}")

  # 4. TODO: Add evaluation step periodically
  # 5. TODO: Add checkpointing periodically (saving state)

end_loop_time = time.time()
print(f"\nFinished {num_train_steps} training steps in {end_loop_time - start_loop_time:.2f}s")

# After the loop, you might want the parameters back on the host CPU
final_params_host = jax.device_get(jax.tree.map(lambda x: x[0], replicated_params))
print("Final parameters retrieved from device 0.")

print("\n" + "="*30 + "\n")



# --- 8.6 Evaluation and Checkpointing (Conceptual) ---
print("--- 8.6 Evaluation & Checkpointing (Conceptual) ---")
# Evaluation:
# - Define an `eval_step` function (can also be pmapped).
# - Takes `params` and an `eval_batch`.
# - Calls `model.apply({'params': params}, ..., train=False)`.
# - Computes metrics (e.g., accuracy).
# - Uses collectives (`psum`, `pmean`) to aggregate metrics across devices if pmapped.
# - Run evaluation periodically (e.g., every N steps or end of epoch).

# Checkpointing:
# - Need to save `params` and `opt_state` to resume training.
# - Get state from device 0: `host_params = jax.device_get(replicated_params[0])`
# - Use libraries like `orbax.checkpoint` (part of Flax ecosystem) for robust saving/loading
#   of potentially large PyTrees, or manually use `pickle`, `numpy.savez`.
# - Save periodically within the training loop.
print("Evaluation and checkpointing are crucial for real training runs.")

# ** PyTorch Contrast **
# - Loop Structure: Similar outer loop, but state updates are often implicit within
#   `optimizer.step()` (in-place) and model state managed by `nn.Module`/DDP.
#   JAX requires explicit state passing (`new_state = train_step(old_state, ...)`).
# - Parallelism State: DDP often manages replication more implicitly. `pmap` requires
#   more explicit handling of replicated state vs. sharded data, often using explicit
#   replication (`device_put_replicated`) and ensuring the state passed back to `pmap`
#   remains replicated.
# - Evaluation/Checkpointing: Concepts are similar, implementation differs based on state
#   representation (`state_dict` vs. PyTree) and preferred libraries (`torch.save` vs. `orbax`).

print("\n" + "="*30 + "\n")


# --- Module 8 Summary ---
# - A JAX/Flax fine-tuning loop combines a Flax model (potentially using pre-trained parts),
#   an Optax optimizer, loss calculation, and gradient computation within a training step function.
# - This `training_step` function is typically JIT-compiled and potentially `pmap`-ed for parallelism.
# - If using `pmap`:
#    - Data must be sharded (split) across devices (leading dimension).
#    - Parameters and optimizer state are typically replicated across devices. Explicit replication
#      using `jax.device_put_replicated` before the loop is common practice.
#    - Cross-device communication (e.g., gradient averaging using `jax.lax.pmean`) must happen
#      explicitly inside the function being `pmap`-ed, using a consistent `axis_name`.
#    - The state passed *into* `pmap` each iteration should be the replicated state from the *previous* step.
# - The Python training loop manages the flow of state (passing replicated state into `pmap`, getting
#   updated replicated state back) and orchestrates data loading, logging, evaluation, and checkpointing.
# - This structure emphasizes functional purity (the `training_step` itself) and explicit state management.

# End of Module 8 & Curriculum