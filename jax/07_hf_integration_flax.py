# 07_hf_integration_flax.py

# Welcome to Module 7 of the JAX Learning Curriculum!
# Objective: Introduce Flax for structuring JAX models and integrate a pre-trained
#            Hugging Face transformer model (using JAX weights) into our workflow.
# Theme Integration: Load a small transformer model suitable for fine-tuning
#                    on RAG/agentic tasks and perform a forward pass.

import jax
import jax.numpy as jnp
import jax.random as random
import flax.linen as nn # Linen is the core Flax module API
import optax # We might need this later if defining loss here
import time

# Make sure transformers is installed: pip install transformers
from transformers import AutoTokenizer, FlaxAutoModel



print("--- 7.1 Why Higher-Level Libraries? Flax Introduction ---")
# Working directly with PyTrees for parameters (like in Module 4) is feasible
# for small models, but quickly becomes complex.
# Libraries like Flax (by Google) and Haiku (by DeepMind) provide abstractions
# similar to `torch.nn.Module` for building complex networks more easily.
# Key Flax Concepts (`flax.linen`):
# - Modules (`nn.Module`): Reusable components (layers, models).
# - Stateless Parameters: By default, Flax modules don't store parameters internally
#   in `self`. Parameters are kept in an external PyTree.
# - `setup()` method: Called once to define submodules/layers. Does NOT access inputs.
# - `__call__()` method (or others): Defines the forward computation logic, taking inputs.
# - `module.init(rng_key, dummy_input)`: Initializes parameters by running `setup`
#   and a dummy forward pass. Returns the parameter PyTree.
# - `module.apply({'params': params}, inputs)`: Runs the forward pass using the
#   *external* parameter PyTree provided.

print("Flax `linen` module API provides structure similar to other frameworks.")

print("\n" + "="*30 + "\n")



# --- 7.2 Defining a Simple Flax Module (MLP Recap) ---
# Let's redefine our 2-layer MLP from Module 4 using Flax.

class FlaxMLP(nn.Module):
  hidden_dim: int
  output_dim: int

  @nn.compact # Recommended decorator for concise setup/call
  def __call__(self, x):
    # Layers are implicitly defined and named here due to @nn.compact
    # Alternatively, define them in a separate setup() method:
    # self.dense1 = nn.Dense(features=self.hidden_dim) ...
    x = nn.Dense(features=self.hidden_dim, name='linear1')(x)
    x = nn.relu(x)
    x = nn.Dense(features=self.output_dim, name='linear2')(x)
    return x

"""
# Alternatively without @nn.compact:

class FlaxMLP(nn.Module):
  hidden_dim: int
  output_dim: int

  def setup(self):
    # Explicitly define the layers here.
    # Flax automatically handles parameter creation behind the scenes.
    self.dense1 = nn.Dense(features=self.hidden_dim, name='linear1')
    self.dense2 = nn.Dense(features=self.output_dim, name='linear2')

  # @nn.compact # Recommended decorator for concise setup/call --> REMOVED
  def __call__(self, x):
    # Use the layers defined in setup()
    x = self.dense1(x)
    x = nn.relu(x)
    x = self.dense2(x)
    return x
"""

# Instantiate the module descriptor
input_dim = 4
hidden_dim_mlp = 8
output_dim_mlp = 2
mlp = FlaxMLP(hidden_dim=hidden_dim_mlp, output_dim=output_dim_mlp)

# Initialize parameters
key = random.PRNGKey(0)
key, init_key = random.split(key)
dummy_mlp_input = jnp.ones([1, input_dim]) # Need dummy input shape for init
mlp_params = mlp.init(init_key, dummy_mlp_input)['params'] # Init returns {'params': {...}} usually

print("--- 7.2 Simple Flax MLP ---")
print(f"Initialized MLP parameter PyTree structure:")
print(jax.tree.map(lambda p: p.shape, mlp_params)) # Note the naming 'linear1', 'linear2'

# Apply the module (run forward pass)
output_mlp = mlp.apply({'params': mlp_params}, dummy_mlp_input)
print(f"\nMLP Input shape: {dummy_mlp_input.shape}")
print(f"MLP Output shape: {output_mlp.shape}")
print(f"MLP Output value: {output_mlp}")

# ** PyTorch Contrast **
# - `FlaxMLP` resembles `torch.nn.Module`.
# - `nn.Dense` is like `torch.nn.Linear`.
# - Key difference: In Flax, `mlp_params` are stored *outside* the `mlp` object.
#   In PyTorch, parameters are attributes (`self.linear1.weight`).
# - Initialization: Flax `init()` vs. PyTorch parameter creation often in `__init__`.
# - Execution: Flax `apply({'params': ...}, ...)` vs. PyTorch `model(inputs)`.

print("\n" + "="*30 + "\n")



# --- 7.3 Loading a Hugging Face JAX/Flax Model ---
# Let's load a small pre-trained transformer model using its Flax version.
# We'll use DistilBERT - a smaller, faster version of BERT.

model_name = "distilbert-base-uncased" # A good small model to start with

print(f"--- 7.3 Loading Hugging Face Model: {model_name} (Flax version) ---")

# 1. Load Tokenizer (same as for PyTorch)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Tokenizer loaded.")

# 2. Load Flax Model
# Use `FlaxAutoModel` or the specific class like `FlaxDistilBertModel`
# Transformers determines the framework based on installed libraries or explicit args.
try:
    # This automatically downloads and caches the Flax weights ('flax_model.msgpack')
    flax_model = FlaxAutoModel.from_pretrained(model_name)
    # Alternative specific class:
    # from transformers import FlaxDistilBertModel
    # flax_model = FlaxDistilBertModel.from_pretrained(model_name)
    print(f"Flax model '{model_name}' loaded successfully.")
except Exception as e:
    print(f"ERROR loading Flax model: {e}")
    print("Ensure Flax is installed (`pip install flax`) and you have internet.")
    print("Sometimes specifying `from_pt=True` might be needed if only PyTorch weights exist and need conversion (requires PyTorch installed).")
    flax_model = None # Set to None if loading fails

# The loaded `flax_model` is a Flax Module instance with parameters already loaded
# and typically stored directly on the object for convenience (though apply needs them passed).
if flax_model:
    print("\nModel Parameter PyTree structure (showing top levels):")
    try:
        # Accessing parameters might vary slightly based on HF version/model
        # Common ways: model.params or accessing submodules like model.body.params
        if hasattr(flax_model, 'params'):
           params_hf = flax_model.params
        else:
           # Fallback or alternative structure inspection might be needed
           print("Model parameters not found directly at `flax_model.params`. Structure might differ.")
           params_hf = None # Cannot proceed without params

        if params_hf:
            # Print only top-level keys and shapes for brevity
            print(jax.tree.map(lambda x: x.shape, params_hf)) # Might be very large/nested
    except Exception as e:
        print(f"Could not inspect model parameters: {e}")
        params_hf = None


print("\n" + "="*30 + "\n")



# --- 7.4 Using the Hugging Face Flax Model ---

if flax_model and params_hf:
    print("--- 7.4 Forward Pass with HF Flax Model ---")

    # Prepare sample input text
    text = "Hello, this is a test sentence."
    print(f"Input text: '{text}'")

    # Tokenize the input
    # `return_tensors='jax'` returns JAX arrays directly!
    inputs = tokenizer(text, return_tensors="jax", padding=True, truncation=True)
    # Apply jax.tree.map to the .data attribute, which is the dictionary holding the arrays
    print(f"\nTokenized input shapes: {jax.tree.map(lambda x: x.shape, inputs.data)}")
    print(f"Tokenized input IDs: {inputs['input_ids']}")

    # Perform a forward pass (inference)
    # For inference, often direct call works, implicitly using stored params.
    # For training/grads, use `apply`. Let's prepare for that.
    print("\nRunning forward pass...")
    start_time = time.time()
    # Call the model instance directly, passing parameters via the 'params' keyword argument
    # This is a common pattern for Hugging Face Flax models.
    outputs = flax_model(input_ids=inputs['input_ids'],
                         attention_mask=inputs['attention_mask'],
                         params=params_hf) # Pass params here
    end_time = time.time()
    print(f"Forward pass completed in {end_time - start_time:.4f}s")

    # Inspect the output
    # HF models return structured outputs (dataclasses or dicts)
    print(f"\nOutput type: {type(outputs)}")
    # Common output fields for base models:
    if hasattr(outputs, 'last_hidden_state'):
      last_hidden_state = outputs.last_hidden_state
      print(f"Output last_hidden_state shape: {last_hidden_state.shape}") # (batch_size, seq_len, hidden_dim)
      print(f"Example output value (first token embedding):\n{last_hidden_state[0, 0, :5]}...") # Print start of first embedding
    else:
      print("Output structure might differ. Available keys:", outputs.keys() if isinstance(outputs, dict) else dir(outputs))

    # ** Integration into Training Step **
    # - The `params_hf` PyTree would replace our simple `mlp_params`.
    # - The `forward_pass_mlp` function would be replaced by logic using `flax_model.apply`.
    # - The loss function would operate on `outputs.last_hidden_state` (or other relevant outputs).
    # - `value_and_grad` would compute gradients w.r.t the `params_hf` PyTree.
    # - Optax would initialize its state based on `params_hf` and apply updates to it.
    # Everything scales up, but the principles remain the same!

    # ** PyTorch Contrast **
    # - Loading: `AutoModel.from_pretrained(model_name)` returns `torch.nn.Module`.
    # - Parameters: Accessed via `model.parameters()` or `model.state_dict()`.
    # - Forward Pass: `outputs = model(**inputs)` (where inputs is dict of torch tensors).
    # - The overall usage pattern is very similar, mainly differing in how parameters
    #   are stored and passed (external PyTree in Flax vs. internal state in PyTorch).

else:
    print("Skipping forward pass due to model loading failure.")

print("\n" + "="*30 + "\n")

# --- Module 7 Summary ---
# - Higher-level libraries like Flax (`flax.linen`) simplify building complex models in JAX,
#   providing an API similar to `torch.nn.Module`.
# - Flax Modules (`nn.Module`) define structure (`setup` or `@nn.compact`) and computation (`__call__`).
# - Parameters are typically stored externally in PyTrees and passed explicitly during computation
#   using `module.apply({'params': params}, inputs)`. Parameters are created via `module.init()`.
# - Hugging Face `transformers` provides Flax versions of many pre-trained models.
# - Load them using `FlaxAutoModel.from_pretrained(model_name)` or specific classes like `FlaxDistilBertModel`.
# - The loaded model is a Flax Module; its parameters (e.g., `model.params`) are a PyTree compatible
#   with JAX transformations (`grad`, `jit`) and Optax.
# - Use the corresponding `AutoTokenizer` with `return_tensors='jax'` to get JAX array inputs.
# - Running inference or integrating into a training step involves using `model.apply` with the parameter PyTree.
# - This allows leveraging powerful pre-trained models within the JAX functional ecosystem.

# End of Module 7