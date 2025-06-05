# -*- coding: utf-8 -*-
# 03_building_networks_torch_nn.py

# Module 3: Building Neural Networks with `torch.nn`
#
# This script covers:
# 1. `torch.nn.Module`: The base class for all network components.
# 2. Defining Custom Models: Subclassing `nn.Module` with `__init__` and `forward`.
# 3. Parameters (`nn.Parameter`) and Buffers: Learnable vs. non-learnable state.
# 4. Common Layers: `nn.Linear`, `nn.ReLU`, `nn.Embedding`, `nn.LayerNorm`, `nn.Dropout`.
# 5. Containers: `nn.Sequential` for simple architectures.
# 6. Weight Initialization (Brief Overview).
# 7. Performing a Forward Pass and Moving Models to Devices.
# 8. Comparisons to JAX/Flax/Haiku model definition paradigms.

import torch
import torch.nn as nn
import torch.nn.functional as F # Often used for stateless operations like activations
from collections import OrderedDict

print("--- Module 3: Building Neural Networks with torch.nn ---\n")
print(f"Using PyTorch version: {torch.__version__}")

# Determine device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA (GPU) is available. Using device: {device}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"MPS is available. Using device: {device}")
else:
    device = torch.device("cpu")
    print(f"No GPU/MPS found. Using device: {device}")


# %% 1. `torch.nn.Module` Basics
print("\n--- 1. `torch.nn.Module` Basics ---\n")
# `nn.Module` is the cornerstone for building networks in PyTorch.
# Key ideas:
# - Modules contain other Modules (layers, subnetworks) forming a tree structure.
# - Modules contain Parameters (`nn.Parameter`), which are learnable tensors (weights, biases).
# - Modules contain Buffers, which are stateful tensors but not learnable (e.g., running mean in BatchNorm).
# - Modules have a `forward` method defining the computation.
# - You typically define layers in the `__init__` method and the computation flow in `forward`.


# %% 2. Defining a Simple Custom Model (MLP)
print("\n--- 2. Defining a Simple Custom Model (MLP) ---\n")

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__() # Essential: Call the parent class's __init__

        # Define layers as attributes. PyTorch automatically registers them.
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU() # Stateful activation layer
        self.layer_2 = nn.Linear(hidden_size, output_size)
        print("SimpleMLP initialized.")

    def forward(self, x):
        # Define the forward pass computation using the layers defined above.
        # Input 'x' shape: (batch_size, input_size)
        print("  Executing SimpleMLP forward pass...")
        x = self.layer_1(x)
        print(f"    Shape after layer_1: {x.shape}")
        x = self.activation(x) # Using the nn.ReLU module instance
        # Alternatively, use the functional form F.relu(x) - more on this later
        print(f"    Shape after activation: {x.shape}")
        x = self.layer_2(x)
        print(f"    Shape after layer_2: {x.shape}")
        # Output 'x' shape: (batch_size, output_size)
        return x

# Instantiate the model
input_dim = 10
hidden_dim = 20
output_dim = 5
model = SimpleMLP(input_dim, hidden_dim, output_dim)

# Print the model structure
print("\nModel Architecture:")
print(model)


# %% 3. Parameters (`nn.Parameter`) and Buffers
print("\n--- 3. Parameters and Buffers ---\n")

# Parameters are Tensors automatically registered when assigned as attributes
# of `nn.Module` or wrapped in `nn.Parameter()`. They implicitly have requires_grad=True.
print("Model Parameters (Name, Shape, requires_grad):")
total_params = 0
for name, param in model.named_parameters():
    print(f"  {name:<15}: {param.shape}, requires_grad={param.requires_grad}")
    total_params += param.numel() # Count number of elements
print(f"Total number of parameters: {total_params}")

# Buffers are tensors that are part of the module's state but are not updated
# by the optimizer during training (e.g., running mean/variance in BatchNorm).
# They are registered using `register_buffer('buffer_name', tensor)`.
class ModelWithBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(5, 5)
        # Register a buffer named 'call_count' initialized to 0
        # persistent=True (default) means it will be saved in state_dict
        self.register_buffer('call_count', torch.tensor(0, dtype=torch.long), persistent=True)

    def forward(self, x):
        # Buffers can be accessed like attributes
        self.call_count += 1 # Modify the buffer (in-place)
        print(f"  ModelWithBuffer called {self.call_count.item()} times.")
        return self.layer(x)

buffer_model = ModelWithBuffer()
print("\nModel with Buffer:")
# Check parameters (only from self.layer)
for name, param in buffer_model.named_parameters():
    print(f"  Param: {name}")
# Check buffers
for name, buf in buffer_model.named_buffers():
    print(f"  Buffer: {name}, Value: {buf.item()}")

# Run forward a few times to see buffer update
dummy_input_buf = torch.randn(1, 5)
buffer_model(dummy_input_buf)
buffer_model(dummy_input_buf)

# `state_dict`: A dictionary containing all parameters and persistent buffers.
# Crucial for saving and loading models (Module 12).
print("\nModel State Dictionary (state_dict):")
state_dict = model.state_dict()
# Convert to regular dict for cleaner printing of keys/shapes
printable_state_dict = {k: v.shape for k, v in state_dict.items()}
print(OrderedDict(printable_state_dict))

print("\nModelWithBuffer State Dictionary:")
print(buffer_model.state_dict())


# %% 4. Common `nn` Layers
print("\n--- 4. Common `nn` Layers ---\n")

# nn.Linear: Fully connected layer (y = xA^T + b)
linear_layer = nn.Linear(in_features=10, out_features=5)
print(f"Linear Layer: {linear_layer}")
dummy_linear_input = torch.randn(4, 10) # (batch_size, in_features)
linear_output = linear_layer(dummy_linear_input)
print(f"  Output shape: {linear_output.shape}") # (batch_size, out_features)

# nn.ReLU: Activation function (applied element-wise)
relu = nn.ReLU()
print(f"\nReLU Layer: {relu}")
relu_output = relu(linear_output)
# Functional alternative: F.relu(linear_output) - often used directly in forward pass
# Pros of nn.ReLU: Explicit in model structure printout.
# Pros of F.relu: Slightly less code in __init__ if no parameters needed.

# nn.Embedding: Lookup table for discrete inputs (e.g., word indices)
vocab_size = 100 # e.g., 100 unique words
embedding_dim = 16
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
print(f"\nEmbedding Layer: {embedding_layer}")
# Input: tensor of indices (LongTensor)
dummy_indices = torch.randint(0, vocab_size, (4, 5)) # (batch_size, sequence_length)
embedding_output = embedding_layer(dummy_indices)
print(f"  Input indices shape: {dummy_indices.shape}")
print(f"  Output embedding shape: {embedding_output.shape}") # (batch_size, sequence_length, embedding_dim)

# nn.LayerNorm: Normalizes features within a layer/embedding
# normalized_shape typically matches the last dimension(s) of the input tensor
layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
print(f"\nLayerNorm Layer: {layer_norm}")
layernorm_output = layer_norm(embedding_output)
print(f"  Output LayerNorm shape: {layernorm_output.shape}") # Same as input

# nn.Dropout: Randomly zeroes elements during training for regularization
# IMPORTANT: Disabled during evaluation via `model.eval()`
dropout = nn.Dropout(p=0.5) # Probability of zeroing an element
print(f"\nDropout Layer: {dropout}")
dropout_output_train = dropout(layernorm_output) # In training mode by default
# To see effect of eval mode:
dropout.eval()
dropout_output_eval = dropout(layernorm_output) # In eval mode, dropout is bypassed
dropout.train() # Set back to train mode
print("  Dropout output shape (train/eval):", dropout_output_train.shape, dropout_output_eval.shape)
# Note: Outputs will be different due to random zeroing in train mode.
# In eval mode, output should equal input. Check if this holds:
# print(torch.allclose(layernorm_output, dropout_output_eval)) # Should be True


# %% 5. Containers: `nn.Sequential`
print("\n--- 5. Containers: `nn.Sequential` ---\n")

# A convenient way to build models consisting of a simple linear stack of layers.
sequential_model = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim)
)

print("Sequential Model Architecture:")
print(sequential_model)

# Forward pass is implicit (applies layers in defined order)
dummy_input_seq = torch.randn(4, input_dim)
seq_output = sequential_model(dummy_input_seq)
print(f"\nSequential Model Output Shape: {seq_output.shape}") # (4, output_dim)

# Pros: Concise for simple stacks.
# Cons: Less flexible than subclassing `nn.Module` for models with skip connections,
#       multiple inputs/outputs, or complex internal logic.


# %% 6. Weight Initialization (Brief Overview)
print("\n--- 6. Weight Initialization (Brief Overview) ---\n")

# PyTorch layers have default initializations (often Kaiming He or Xavier Glorot).
print("Default initial weights (example from sequential_model layer 0):")
print(sequential_model[0].weight.data[0, :5]) # Show first 5 weights of first neuron

# Custom initialization can be done, e.g., by iterating parameters or using `model.apply()`
def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # For every Linear layer in a model
    if classname.find('Linear') != -1:
        # Apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(0.0, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0)

# Create a new model instance
model_to_init = SimpleMLP(input_dim, hidden_dim, output_dim)
# Apply the custom initialization function recursively to all submodules
model_to_init.apply(weights_init_uniform_rule)
print("\nCustom initialized weights (uniform 0-1, example from layer_1):")
print(model_to_init.layer_1.weight.data[0, :5])

# Initialization is important but complex; often defaults are good starting points.


# %% 7. Forward Pass & Device Management for Models
print("\n--- 7. Forward Pass & Device Management for Models ---\n")

# Move the model to the target device (GPU/MPS/CPU)
# NOTE: `model.to(device)` modifies the model object in-place regarding its parameters/buffers device.
print(f"Moving model to device: {device}")
model.to(device)

# Check a parameter's device after moving
print(f"Device of layer_1 weight after move: {model.layer_1.weight.device}")

# Create dummy input data *on the same device*
batch_size = 4
input_data = torch.randn(batch_size, input_dim, device=device)
print(f"\nInput data shape: {input_data.shape}, device: {input_data.device}")

# Perform the forward pass
# Ensure model is in the correct mode (train vs eval) if using Dropout/BatchNorm
model.eval() # Set to evaluation mode (disables dropout, uses running stats in BN)
# model.train() # Set back to training mode if needed

# It's good practice to use torch.no_grad() during inference
with torch.no_grad():
    output = model(input_data)

print(f"\nOutput data shape: {output.shape}, device: {output.device}")
model.train() # Set back to training mode for consistency if subsequent training occurs


# %% 8. JAX Comparison
print("\n--- 8. JAX Comparison ---\n")

# Key Differences Recap:
# - PyTorch: Stateful `nn.Module` objects encapsulate parameters and code. Parameters are implicitly found via `model.parameters()`.
# - JAX: Typically stateless functions. Parameters live externally (e.g., in PyTrees like dicts/lists) and are passed explicitly to functions.
#
# Libraries like Flax and Haiku provide `nn.Module`-like abstractions for JAX:
# - Flax `linen.Module`: Uses `setup()` (like `__init__`) to define layers/submodules and `__call__` (like `forward`). Parameters are managed implicitly within the module during the *first* call (or via an explicit `init` call) but are still returned externally and passed explicitly in subsequent calls.
# - Haiku: Uses `hk.transform` to convert functions containing object-oriented module definitions into pure function pairs (`init`, `apply`) that manage state explicitly.
#
# Example JAX/Flax Idea (Conceptual):
# ```python
# # Flax Example Sketch (Not runnable here)
# import flax.linen as nn
# import jax
# import jax.numpy as jnp
#
# class FlaxMLP(nn.Module):
#     hidden_size: int
#     output_size: int
#
#     @nn.compact # Decorator allows defining submodules inline in __call__ or setup
#     def __call__(self, x):
#         x = nn.Dense(features=self.hidden_size)(x)
#         x = nn.relu(x) # Functional activation
#         x = nn.Dense(features=self.output_size)(x)
#         return x
#
# key = jax.random.PRNGKey(0)
# input_shape = (1, input_dim) # Batch dim typically needed for init
# dummy_input = jnp.ones(input_shape)
#
# flax_model = FlaxMLP(hidden_size=hidden_dim, output_size=output_dim)
# params = flax_model.init(key, dummy_input)['params'] # Initialize and get parameters
# print("Flax initialized parameters (structure):", jax.tree_map(lambda x: x.shape, params))
#
# # Forward pass requires explicit parameters
# output_flax = flax_model.apply({'params': params}, dummy_input)
# ```
# Parameter Initialization: JAX requires explicit PRNG keys for randomness.
# State Management: More explicit handling of parameters and potentially other state (like BatchNorm stats) is required in JAX frameworks compared to PyTorch's integrated `nn.Module` state.


# %% Conclusion
print("\n--- Module 3 Summary ---\n")
print("Key Takeaways:")
print("- `nn.Module` is the base for building networks in PyTorch.")
print("- Define layers in `__init__` and computation in `forward`.")
print("- Parameters (`nn.Parameter`) are automatically tracked learnable tensors.")
print("- Buffers store non-learnable state.")
print("- Use built-in layers (`nn.Linear`, `nn.ReLU`, etc.) or `nn.Sequential`.")
print("- Models and data must be on the same device (`model.to(device)`).")
print("- PyTorch's stateful approach contrasts with JAX's typically stateless, functional approach requiring explicit parameter handling.")

print("\nEnd of Module 3.")