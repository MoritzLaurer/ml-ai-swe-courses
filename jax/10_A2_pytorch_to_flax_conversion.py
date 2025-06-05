# A2_pytorch_to_flax_conversion_transformer.py

# Welcome to Advanced Module A.2 (Transformer Version)!
# Objective: Understand the process of translating a Transformer-like model
#            architecture from PyTorch to Flax and loading PyTorch weights.
# Theme Integration: Convert weights for a simplified Transformer Encoder block.

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F
import jax.random as random
from flax.core.frozen_dict import unfreeze
from typing import Optional
import flax



# --- A2.1 Defining Equivalent Transformer Encoder Blocks ---
print("--- A2.1 Defining Equivalent Transformer Encoder Blocks ---")

# --- Configuration ---
embed_dim = 64 # Dimension of embeddings
num_heads = 4  # Number of attention heads
ff_dim = 128   # Dimension of the feed-forward layer
dropout_rate = 0.1 # Dropout rate (will be disabled for verification)

# --- PyTorch Model Definition ---
class TransformerEncoderLayerPT(tnn.Module):
    """Simplified Pre-Norm Transformer Encoder Layer (PyTorch)."""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.norm1 = tnn.LayerNorm(embed_dim)
        self.mha = tnn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.dropout1 = tnn.Dropout(dropout_rate)

        self.norm2 = tnn.LayerNorm(embed_dim)
        self.linear1 = tnn.Linear(embed_dim, ff_dim)
        self.relu = tnn.ReLU()
        self.dropout2 = tnn.Dropout(dropout_rate)
        self.linear2 = tnn.Linear(ff_dim, embed_dim)
        self.dropout3 = tnn.Dropout(dropout_rate)

    def forward(self, x, src_mask: Optional[torch.Tensor] = None):
        # Pre-Norm Structure
        # Attention Block
        residual = x
        x_norm = self.norm1(x)
        # Note: MHA expects query, key, value. For self-attention, they are the same.
        attn_output, _ = self.mha(x_norm, x_norm, x_norm, attn_mask=src_mask, need_weights=False)
        x = residual + self.dropout1(attn_output)

        # Feed-Forward Block
        residual = x
        x_norm = self.norm2(x)
        x_ff = self.linear2(self.dropout2(self.relu(self.linear1(x_norm))))
        x = residual + self.dropout3(x_ff)
        return x

# Instantiate PyTorch model
pytorch_model = TransformerEncoderLayerPT(embed_dim, num_heads, ff_dim, dropout_rate)
pytorch_model.eval() # Set to eval mode to disable dropout for weight loading/verification
pytorch_state_dict = pytorch_model.state_dict()
print("PyTorch Model Instantiated.")
print(f"PyTorch state_dict keys: {list(pytorch_state_dict.keys())}")



# --- Flax Model Definition ---
class TransformerEncoderLayerFlax(nn.Module):
    """Simplified Pre-Norm Transformer Encoder Layer (Flax)."""
    embed_dim: int
    num_heads: int
    ff_dim: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, mask: Optional[jnp.ndarray] = None, deterministic: bool = True):
        # --- Attention Block ---
        residual = x
        x_norm = nn.LayerNorm(name='norm1')(x)

        # Use SelfAttention which bundles MHA logic for self-attention case
        # Or use MultiHeadDotProductAttention and pass x_norm 3 times.
        # We use SelfAttention for conciseness here. It uses Dense layers internally.
        attn_output = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim, # Often same as input embed_dim
            dropout_rate=self.dropout_rate,
            deterministic=deterministic, # Control dropout
            name='mha'
        )(x_norm, mask=mask) # Flax attention masks are often additive (-inf for masked) or boolean

        x = residual + nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(attn_output)

        # --- Feed-Forward Block ---
        residual = x
        x_norm = nn.LayerNorm(name='norm2')(x)

        # Flax uses nn.Dense for linear layers
        x_ff = nn.Dense(features=self.ff_dim, name='linear1')(x_norm)
        x_ff = nn.relu(x_ff)
        x_ff = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x_ff)
        x_ff = nn.Dense(features=self.embed_dim, name='linear2')(x_ff) # Project back to embed_dim
        x_ff = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x_ff)

        x = residual + x_ff
        return x


# Initialize Flax model to get parameter structure
key = random.PRNGKey(1)
dummy_flax_input_shape = (1, 16, embed_dim) # (Batch, SeqLen, EmbedDim)
dummy_flax_input = jnp.ones(dummy_flax_input_shape)
flax_model = TransformerEncoderLayerFlax(embed_dim, num_heads, ff_dim, dropout_rate)
# Pass deterministic=True during init and apply for verification
flax_params_init = flax_model.init(key, dummy_flax_input, deterministic=True)['params']

print("\nFlax Model Initial Parameter Structure:")
# Use pprint for better nested dict formatting if needed
from pprint import pprint
pprint(jax.tree.map(lambda p: (p.shape, p.dtype), flax_params_init))

print("\n" + "="*30 + "\n")



# --- A2.2 Weight Conversion: Focus on Attention ---
print("--- A2.2 Weight Conversion ---")
# The most complex part is mapping PyTorch's MHA weights to Flax's internal Dense layers.

# Create a modifiable copy of the initialized Flax params structure
flax_params_converted = unfreeze(flax_params_init)

# --- Conversion Logic ---

# **LayerNorms (norm1, norm2)**
# PyTorch: weight, bias -> Flax: scale, bias
flax_params_converted['norm1']['scale'] = pytorch_state_dict['norm1.weight'].detach().cpu().numpy()
flax_params_converted['norm1']['bias'] = pytorch_state_dict['norm1.bias'].detach().cpu().numpy()
flax_params_converted['norm2']['scale'] = pytorch_state_dict['norm2.weight'].detach().cpu().numpy()
flax_params_converted['norm2']['bias'] = pytorch_state_dict['norm2.bias'].detach().cpu().numpy()
print("Converted norm1, norm2 weights/biases.")

# **Multi-Head Attention (mha)**
# PyTorch combines Q, K, V projections into 'in_proj_weight' [3*embed_dim, embed_dim]
# and potentially 'in_proj_bias' [3*embed_dim].
# Flax `SelfAttention` (or `MultiHeadDotProductAttention`) uses separate Dense layers internally
# named 'query', 'key', 'value', 'out'. Each has 'kernel' [embed_dim, embed_dim] and 'bias' [embed_dim].
pt_in_proj_w = pytorch_state_dict['mha.in_proj_weight'].detach().cpu().numpy()
pt_in_proj_b = pytorch_state_dict['mha.in_proj_bias'].detach().cpu().numpy()
pt_out_proj_w = pytorch_state_dict['mha.out_proj.weight'].detach().cpu().numpy()
pt_out_proj_b = pytorch_state_dict['mha.out_proj.bias'].detach().cpu().numpy()

# Split the combined input projection weight/bias
q_w, k_w, v_w = np.split(pt_in_proj_w, 3, axis=0)
q_b, k_b, v_b = np.split(pt_in_proj_b, 3, axis=0)

# Assign to Flax structure, transposing weights (O, I -> I, O for Dense kernel)
# AND reshaping to match Flax's internal QKV projection structure (num_heads, head_dim)
head_dim = embed_dim // num_heads # Should be 16

# Reshape Kernels: (embed_dim, embed_dim) -> transpose -> (embed_dim, embed_dim) -> reshape -> (embed_dim, num_heads, head_dim)
flax_params_converted['mha']['query']['kernel'] = np.transpose(q_w).reshape(embed_dim, num_heads, head_dim)
flax_params_converted['mha']['key']['kernel'] = np.transpose(k_w).reshape(embed_dim, num_heads, head_dim)
flax_params_converted['mha']['value']['kernel'] = np.transpose(v_w).reshape(embed_dim, num_heads, head_dim)

# Reshape Biases: (embed_dim,) -> reshape -> (num_heads, head_dim)
flax_params_converted['mha']['query']['bias'] = q_b.reshape(num_heads, head_dim)
flax_params_converted['mha']['key']['bias'] = k_b.reshape(num_heads, head_dim)
flax_params_converted['mha']['value']['bias'] = v_b.reshape(num_heads, head_dim)

# Reshape Output Kernel: (embed_dim, embed_dim) -> transpose -> (embed_dim, embed_dim) -> reshape -> (num_heads, head_dim, embed_dim)
flax_params_converted['mha']['out']['kernel'] = np.transpose(pt_out_proj_w).reshape(num_heads, head_dim, embed_dim)
# Output bias remains (embed_dim,)
flax_params_converted['mha']['out']['bias'] = pt_out_proj_b
print("Converted mha weights/biases (split in_proj, transposed & reshaped kernels/biases).")

# **Feed-Forward Network (linear1, linear2)**
# Standard Dense layer conversion (transpose kernel)
flax_params_converted['linear1']['kernel'] = np.transpose(pytorch_state_dict['linear1.weight'].detach().cpu().numpy())
flax_params_converted['linear1']['bias'] = pytorch_state_dict['linear1.bias'].detach().cpu().numpy()
flax_params_converted['linear2']['kernel'] = np.transpose(pytorch_state_dict['linear2.weight'].detach().cpu().numpy())
flax_params_converted['linear2']['bias'] = pytorch_state_dict['linear2.bias'].detach().cpu().numpy()
print("Converted linear1, linear2 weights/biases.")


# --- Pitfalls Recap Specific to Transformers ---
# 1. Attention Projections: The biggest difference is often how Q, K, V projections are handled
#    (combined in PyTorch `in_proj_weight` vs. separate Dense layers in Flax). Requires splitting/reshaping.
# 2. Layer Naming: Ensure PyTorch names (`mha`, `norm1`, etc.) map to Flax names (`mha`, `norm1`). Internal Flax names ('query', 'key', 'value', 'out') are usually fixed within `SelfAttention`.
# 3. LayerNorm Mapping: `weight` -> `scale`, `bias` -> `bias`.
# 4. Dense/Linear Kernel Transpose: Always required.
# 5. Bias Handling: Check if biases are used consistently (`bias=True` in PyTorch MHA/Linear).
# 6. Attention Mask: PyTorch MHA expects `attn_mask` where `True` indicates *ignore*. Flax attention masks are often additive (add `-inf` to positions to ignore) or boolean where `True` indicates *keep*. This doesn't affect weight conversion but matters for running the model.

flax_params_converted = flax.core.freeze(flax_params_converted)
print("\nConverted weights loaded into Flax parameter structure.")

print("\n" + "="*30 + "\n")



# --- A2.3 Verification ---
print("--- A2.3 Verification ---")
# Check if outputs match for the same input, disabling dropout.

# 1. Prepare Input Data
key, input_key = random.split(key)
input_data_np = random.normal(input_key, dummy_flax_input_shape) # (B, S, E)

input_jax = jnp.array(input_data_np)
input_torch = torch.tensor(input_data_np, dtype=torch.float32)

print(f"Input shape for both frameworks: {input_jax.shape}")

# 2. Run PyTorch Inference (ensure model is in eval mode)
pytorch_model.eval()
with torch.no_grad():
    output_torch = pytorch_model(input_torch)
output_torch_np = output_torch.detach().cpu().numpy()
print(f"\nPyTorch Output shape: {output_torch_np.shape}")
print(f"PyTorch Output sample (batch 0, seq 0, first 5 features):\n{output_torch_np[0, 0, :5]}")

# 3. Run Flax Inference (ensure deterministic=True)
# Use the *converted* parameters
output_flax = flax_model.apply(
    {'params': flax_params_converted},
    input_jax,
    deterministic=True # Disable dropout
)
output_flax_np = np.array(output_flax)
print(f"\nFlax Output shape: {output_flax_np.shape}")
print(f"Flax Output sample (batch 0, seq 0, first 5 features):\n{output_flax_np[0, 0, :5]}")

# 4. Compare Outputs
try:
    np.testing.assert_allclose(output_torch_np, output_flax_np, rtol=1e-5, atol=1e-5)
    print("\nSUCCESS: Outputs are numerically close!")
except AssertionError as e:
    print("\nFAILURE: Outputs differ significantly.")
    print(e)
    # Optional: Print difference stats
    # diff = np.abs(output_torch_np - output_flax_np)
    # print(f"Max difference: {np.max(diff)}")
    # print(f"Mean difference: {np.mean(diff)}")

print("\n" + "="*30 + "\n")



# --- Module A2 (Transformer Version) Summary ---
# - Converting Transformer models (like BERT/Llama components) involves careful mapping
#   between PyTorch's `nn.Module` structure and Flax's `linen.Module` structure.
# - Key conversion points for Transformer blocks:
#   - **Multi-Head Attention:** PyTorch often combines Q/K/V projections (`in_proj_weight`).
#     Flax often uses separate internal Dense layers (`query`, `key`, `value`). Requires
#     splitting the PyTorch weight tensor and transposing each part for the Flax kernels.
#     The output projection (`out_proj`) kernel also needs transposing.
#   - **LayerNorm:** Map PyTorch `weight`/`bias` to Flax `scale`/`bias`.
#   - **Feed-Forward Network (MLP):** Standard Linear/Dense conversion applies (transpose kernel).
#   - **Dropout:** Ensure consistency, disable during verification/inference (`model.eval()`, `deterministic=True`).
# - Name mapping between the PyTorch `state_dict` and the Flax parameter PyTree remains crucial.
# - Verification by comparing outputs with dropout disabled is essential. Small numerical
#   differences are expected, use `np.allclose` with tolerance.

# End of Module A2 (Transformer Version)