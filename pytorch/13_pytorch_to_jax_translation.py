# -*- coding: utf-8 -*-
# 13_pytorch_to_jax_translation.py

# Module 13: Translating PyTorch Models and Weights to JAX/Flax
#
# This script demonstrates the process of taking a PyTorch model and its
# trained weights (from Module 8) and translating them for use in JAX/Flax.
# This is a common task when needing to leverage models across different
# ecosystems or for deployment on specific JAX-optimized hardware.
#
# We will cover:
# 1. Key Framework Differences (PyTorch vs. JAX/Flax Recap).
# 2. Loading the Pre-trained PyTorch Model and Weights.
# 3. Defining the Equivalent JAX/Flax Model Architecture.
# 4. Detailed Weight Mapping and Conversion Strategy.
#    - Handling nn.Linear, nn.Embedding, nn.LayerNorm.
#    - Deconstructing PyTorch's nn.MultiheadAttention weights for Flax.
# 5. Initializing the Flax Model with Ported Weights.
# 6. Implementing and Comparing Inference in PyTorch and JAX/Flax.

import torch
import torch.nn as nn
# PyTorch model definition (copied from Module 8 for self-containment)

import jax
import jax.numpy as jnp
import flax.linen as f_nn # Using f_nn alias to avoid clash with torch.nn
import flax
import numpy as np
from transformers import AutoTokenizer
import os
import random

print("--- Module 13: Translating PyTorch Models and Weights to JAX/Flax ---\n")
print(f"Using PyTorch version: {torch.__version__}")
print(f"Using JAX version: {jax.__version__}")
print(f"Using Flax version: {flax.__version__}")

# --- Reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# JAX PRNG keys are handled explicitly.

# Determine device for PyTorch (JAX uses its own device management)
if torch.cuda.is_available():
    pt_device = torch.device("cuda")
    print(f"PyTorch CUDA (GPU) is available. Using device: {pt_device}")
elif torch.backends.mps.is_available():
    pt_device = torch.device("mps") # PyTorch MPS for Apple Silicon
    print(f"PyTorch MPS is available. Using device: {pt_device}")
else:
    pt_device = torch.device("cpu")
    print(f"PyTorch using CPU: {pt_device}")

# JAX device
jax_device = jax.devices()[0]
print(f"JAX is using device: {jax_device}")


# %% 1. Key Framework Differences (PyTorch vs. JAX/Flax Recap)
print("\n--- 1. Key Framework Differences (Recap) ---\n")
print("PyTorch:")
print("  - Stateful `nn.Module` objects encapsulate parameters and code.")
print("  - Parameters are implicitly managed within modules.")
print("  - Eager execution by default.")
print("JAX/Flax:")
print("  - Typically stateless functions for model logic.")
print("  - Flax `linen.Module` helps define architecture but parameters are external (PyTrees).")
print("  - Parameters are explicitly passed to model's `apply` method.")
print("  - JIT compilation (`@jax.jit`) is common for performance.")
print("This difference in state and parameter handling is central to the translation process.")


# %% 2. Loading the Pre-trained PyTorch Model and Weights
print("\n--- 2. Loading the Pre-trained PyTorch Model and Weights ---\n")

# --- PyTorch Model Definitions (from Module 8) ---
class PTTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim_multiplier=4, dropout_rate=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        ff_dim = embed_dim * ff_dim_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout_rate) # Dropout is part of FFN
        )
        self.dropout = nn.Dropout(dropout_rate) # This dropout is for after attention output

    def forward(self, x):
        _batch_size, seq_len, _embed_dim = x.shape
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            sz=seq_len, device=x.device, dtype=x.dtype # Match input dtype
        ).to(x.device) # Ensure mask is on the same device

        norm_x = self.ln1(x)
        attn_output, _ = self.attn(norm_x, norm_x, norm_x,
                                   attn_mask=causal_mask,
                                   is_causal=False, # Explicit mask handles causality
                                   need_weights=False)
        x = x + self.dropout(attn_output) # Dropout after attention sublayer output

        norm_x = self.ln2(x)
        ff_output = self.ffn(norm_x)
        x = x + ff_output # Residual connection for FFN
        return x

class PTSmallGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, block_size, dropout_rate):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(block_size, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.transformer_blocks = nn.ModuleList(
            [PTTransformerBlock(embed_dim, num_heads, dropout_rate=dropout_rate) for _ in range(num_layers)]
        )
        self.ln_final = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        # Enable weight tying
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, idx_sequences):
        _batch_size, seq_len = idx_sequences.shape
        tok_emb = self.token_embedding(idx_sequences)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=idx_sequences.device)
        pos_emb = self.positional_embedding(positions)
        x = tok_emb + pos_emb
        x = self.dropout(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits

# --- Load Checkpoint ---
# IMPORTANT: Update this path to an actual checkpoint from 08_full_gpt_model_train.py
# The name includes a timestamp, so it will vary.
PT_CHECKPOINT_PATH = './checkpoints/run_20250515_105608_E1_S20000_D128H8L6/checkpoint_step_54000.pth' # <--- USER MUST UPDATE THIS

pt_model_instance = None
pt_model_config = None

if os.path.exists(PT_CHECKPOINT_PATH):
    print(f"Loading PyTorch checkpoint from {PT_CHECKPOINT_PATH}...")
    checkpoint = torch.load(PT_CHECKPOINT_PATH, map_location=pt_device)
    pt_model_config = checkpoint['model_config']

    pt_model_instance = PTSmallGPT(
        vocab_size=pt_model_config['vocab_size'],
        embed_dim=pt_model_config['embed_dim'],
        num_heads=pt_model_config['num_heads'],
        num_layers=pt_model_config['num_layers'],
        block_size=pt_model_config['block_size'],
        dropout_rate=pt_model_config['dropout_rate'] # Use dropout from config for consistency
    )
    pt_model_instance.load_state_dict(checkpoint['model_state_dict'])
    pt_model_instance.to(pt_device)
    pt_model_instance.eval() # Set to evaluation mode
    print("PyTorch model loaded and in eval mode.")
    print(f"PyTorch Model Config: {pt_model_config}")
else:
    print(f"PyTorch checkpoint '{PT_CHECKPOINT_PATH}' not found. Cannot proceed with translation.")
    exit()

# Load Tokenizer (should match training)
tokenizer = AutoTokenizer.from_pretrained("gpt2")


# %% 3. Defining the Equivalent JAX/Flax Model Architecture
print("\n--- 3. Defining the Equivalent JAX/Flax Model Architecture ---\n")

class ScaledDotProductAttentionFlax(f_nn.Module):
    dropout_rate: float
    deterministic: bool # To control dropout during inference

    @f_nn.compact
    def __call__(self, query, key, value, mask=None):
        embed_dim = query.shape[-1]
        scores = jnp.matmul(query, jnp.swapaxes(key, -2, -1)) / jnp.sqrt(embed_dim)
        if mask is not None:
            # Flax masks typically are True where valid, False where masked
            # So we want to add a large negative number where mask is False
            scores = jnp.where(mask, scores, -1e9) # Large negative number for masked positions
        
        attn_weights = jax.nn.softmax(scores, axis=-1)
        
        if self.dropout_rate > 0. and not self.deterministic:
            key_dropout = jax.random.fold_in(self.make_rng('dropout'), jax.lax.axis_index('batch'))
            attn_weights = f_nn.Dropout(rate=self.dropout_rate)(attn_weights, deterministic=self.deterministic, rng=key_dropout)

        return jnp.matmul(attn_weights, value)


class MultiHeadAttentionFlax(f_nn.Module):
    embed_dim: int
    num_heads: int
    dropout_rate: float
    deterministic: bool

    def setup(self):
        head_dim = self.embed_dim // self.num_heads
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        # Dense layers for Q, K, V, and output projection
        self.query_proj = f_nn.Dense(features=self.embed_dim, name="query_proj")
        self.key_proj   = f_nn.Dense(features=self.embed_dim, name="key_proj")
        self.value_proj = f_nn.Dense(features=self.embed_dim, name="value_proj")
        self.out_proj   = f_nn.Dense(features=self.embed_dim, name="out_proj")
        self.attention_fn = ScaledDotProductAttentionFlax(dropout_rate=self.dropout_rate, deterministic=self.deterministic, name="scaled_dot_product_attention")


    def __call__(self, x, causal_mask_flax):
        batch_size, seq_len, _ = x.shape
        
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        # Reshape for multi-head: (batch, seq_len, num_heads, head_dim)
        # Then transpose to: (batch, num_heads, seq_len, head_dim)
        def split_heads(tensor):
            return tensor.reshape(batch_size, seq_len, self.num_heads, self.embed_dim // self.num_heads).transpose((0, 2, 1, 3))

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)
        
        # Apply attention, causal_mask_flax from f_nn.make_causal_mask
        # is already (batch, 1, seq_len, seq_len) and suitable for broadcasting.
        # The following line was removed:
        # if causal_mask_flax is not None:
        #      causal_mask_flax = causal_mask_flax[:, None, :, :] 

        attn_output = self.attention_fn(q, k, v, mask=causal_mask_flax)

        # Concatenate heads: (batch, seq_len, embed_dim)
        # attn_output shape should now be (batch, num_heads, seq_len, head_dim)
        attn_output = attn_output.transpose((0, 2, 1, 3)).reshape(batch_size, seq_len, self.embed_dim)
        
        output = self.out_proj(attn_output)
        return output


class FlaxTransformerBlock(f_nn.Module):
    embed_dim: int
    num_heads: int
    dropout_rate: float
    deterministic: bool # For dropout control

    def setup(self):
        self.ln1 = f_nn.LayerNorm(name="ln1")
        self.attn = MultiHeadAttentionFlax(embed_dim=self.embed_dim,
                                           num_heads=self.num_heads,
                                           dropout_rate=self.dropout_rate,
                                           deterministic=self.deterministic,
                                           name="attn")
        self.ln2 = f_nn.LayerNorm(name="ln2")
        ff_dim = self.embed_dim * 4 # Matching PyTorch FFN multiplier
        self.ffn_linear1 = f_nn.Dense(features=ff_dim, name="ffn_linear1")
        self.ffn_linear2 = f_nn.Dense(features=self.embed_dim, name="ffn_linear2")
        
        self.dropout_attn = f_nn.Dropout(rate=self.dropout_rate, name="dropout_attn")
        # self.dropout_ffn is defined but will not be used directly on ff_output if PT model is corrected
        self.dropout_ffn = f_nn.Dropout(rate=self.dropout_rate, name="dropout_ffn") 
        self.dropout_ffn_internal = f_nn.Dropout(rate=self.dropout_rate, name="dropout_ffn_internal")

    def __call__(self, x, causal_mask_flax):
        # Attention block
        norm_x = self.ln1(x)
        attn_output = self.attn(norm_x, causal_mask_flax=causal_mask_flax)
        x = x + self.dropout_attn(attn_output, deterministic=self.deterministic) # Dropout after attention sublayer output

        # FFN block
        norm_x = self.ln2(x)
        ff_hidden = f_nn.relu(self.ffn_linear1(norm_x))
        ff_hidden_dropped = self.dropout_ffn_internal(ff_hidden, deterministic=self.deterministic) # Corresponds to PT FFN internal dropout
        ff_output = self.ffn_linear2(ff_hidden_dropped)
        # Corrected: ff_output already has internal dropout. No additional self.dropout_ffn(ff_output, ...)
        x = x + ff_output 
        return x

class FlaxSmallGPT(f_nn.Module):
    vocab_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    block_size: int
    dropout_rate: float
    deterministic: bool # For dropout control

    def setup(self):
        self.token_embedding = f_nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim, name="token_embedding")
        self.positional_embedding = f_nn.Embed(num_embeddings=self.block_size, features=self.embed_dim, name="positional_embedding")
        self.dropout_emb = f_nn.Dropout(rate=self.dropout_rate, name="dropout_emb")
        
        self.transformer_blocks = [
            FlaxTransformerBlock(embed_dim=self.embed_dim,
                                 num_heads=self.num_heads,
                                 dropout_rate=self.dropout_rate,
                                 deterministic=self.deterministic,
                                 name=f"transformer_blocks_{i}")
            for i in range(self.num_layers)
        ]
        self.ln_final = f_nn.LayerNorm(name="ln_final")
        self.lm_head = f_nn.Dense(features=self.vocab_size, name="lm_head")

    def __call__(self, idx_sequences):
        seq_len = idx_sequences.shape[1]
        
        tok_emb = self.token_embedding(idx_sequences)
        positions = jnp.arange(0, seq_len, dtype=jnp.int32)
        pos_emb = self.positional_embedding(positions)
        
        x = tok_emb + pos_emb
        x = self.dropout_emb(x, deterministic=self.deterministic)

        # Flax causal mask: True for positions to attend to.
        # Shape (batch_size, seq_len, seq_len) or (seq_len, seq_len)
        causal_mask_flax = f_nn.make_causal_mask(idx_sequences, dtype=jnp.bool_)
        
        for block in self.transformer_blocks:
            x = block(x, causal_mask_flax=causal_mask_flax)
            
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits

print("JAX/Flax model definitions (FlaxSmallGPT, FlaxTransformerBlock, etc.) created.")


# %% 4. Detailed Weight Mapping and Conversion Strategy
print("\n--- 4. Detailed Weight Mapping and Conversion Strategy ---\n")

def convert_pytorch_weights_to_flax(pt_state_dict, pt_config):
    """
    Converts PyTorch state_dict to a Flax-compatible parameter PyTree.
    This is highly specific to the model architectures.
    """
    flax_params = {}

    # Token and Positional Embeddings
    flax_params['token_embedding'] = {'embedding': pt_state_dict['token_embedding.weight'].cpu().numpy()}
    flax_params['positional_embedding'] = {'embedding': pt_state_dict['positional_embedding.weight'].cpu().numpy()}

    # Transformer Blocks
    for i in range(pt_config['num_layers']):
        block_key_pt = f'transformer_blocks.{i}'
        block_key_flax = f'transformer_blocks_{i}' # Matches Flax model naming
        flax_params[block_key_flax] = {}

        # LayerNorm1 (ln1)
        flax_params[block_key_flax]['ln1'] = {
            'scale': pt_state_dict[f'{block_key_pt}.ln1.weight'].cpu().numpy(),
            'bias': pt_state_dict[f'{block_key_pt}.ln1.bias'].cpu().numpy()
        }
        
        # Attention (attn) - This is the most complex part
        # PyTorch nn.MultiheadAttention combines Q,K,V input projections.
        # in_proj_weight shape: (3 * embed_dim, embed_dim)
        # in_proj_bias shape: (3 * embed_dim)
        # These need to be split for Flax's separate Dense layers for Q, K, V.
        
        # --- Q, K, V Input Projections ---
        # PT: single in_proj_weight and in_proj_bias
        # Flax: separate query_proj, key_proj, value_proj Dense layers
        pt_in_proj_weight = pt_state_dict[f'{block_key_pt}.attn.in_proj_weight'].cpu().numpy()
        pt_in_proj_bias = pt_state_dict[f'{block_key_pt}.attn.in_proj_bias'].cpu().numpy()

        # Split them (Q, K, V are stacked in this order by PyTorch)
        q_w, k_w, v_w = np.split(pt_in_proj_weight, 3, axis=0)
        q_b, k_b, v_b = np.split(pt_in_proj_bias, 3, axis=0)

        # Flax Dense layers expect kernel shape (in_features, out_features)
        # PyTorch Linear weights are (out_features, in_features), so transpose.
        flax_params[block_key_flax]['attn'] = {
            'query_proj': {'kernel': q_w.T, 'bias': q_b},
            'key_proj':   {'kernel': k_w.T, 'bias': k_b},
            'value_proj': {'kernel': v_w.T, 'bias': v_b}
        }
        
        # --- Attention Output Projection ---
        # PT: attn.out_proj.weight and attn.out_proj.bias
        # Flax: out_proj Dense layer
        flax_params[block_key_flax]['attn']['out_proj'] = {
            'kernel': pt_state_dict[f'{block_key_pt}.attn.out_proj.weight'].cpu().numpy().T, # Transpose
            'bias': pt_state_dict[f'{block_key_pt}.attn.out_proj.bias'].cpu().numpy()
        }
        # Note: The ScaledDotProductAttentionFlax itself has no learnable params here.

        # LayerNorm2 (ln2)
        flax_params[block_key_flax]['ln2'] = {
            'scale': pt_state_dict[f'{block_key_pt}.ln2.weight'].cpu().numpy(),
            'bias': pt_state_dict[f'{block_key_pt}.ln2.bias'].cpu().numpy()
        }

        # FFN
        # PT: ffn.0 (linear1), ffn.2 (linear2)
        # Flax: ffn_linear1, ffn_linear2
        flax_params[block_key_flax]['ffn_linear1'] = {
            'kernel': pt_state_dict[f'{block_key_pt}.ffn.0.weight'].cpu().numpy().T, # Transpose
            'bias': pt_state_dict[f'{block_key_pt}.ffn.0.bias'].cpu().numpy()
        }
        flax_params[block_key_flax]['ffn_linear2'] = {
            'kernel': pt_state_dict[f'{block_key_pt}.ffn.2.weight'].cpu().numpy().T, # Transpose
            'bias': pt_state_dict[f'{block_key_pt}.ffn.2.bias'].cpu().numpy()
        }
        # Dropout layers in Flax (dropout_attn, dropout_ffn, dropout_ffn_internal, dropout_emb) do not have parameters.

    # Final LayerNorm (ln_final)
    flax_params['ln_final'] = {
        'scale': pt_state_dict['ln_final.weight'].cpu().numpy(),
        'bias': pt_state_dict['ln_final.bias'].cpu().numpy()
    }

    # Language Model Head (lm_head)
    flax_params['lm_head'] = {
        'kernel': pt_state_dict['lm_head.weight'].cpu().numpy().T, # Transpose
        'bias': pt_state_dict['lm_head.bias'].cpu().numpy()
    }
    
    # Dropout layers (dropout_emb for FlaxSmallGPT) don't have weights.
    return {'params': flax_params} # Flax expects params to be under a 'params' key typically

# Perform the conversion if PyTorch model was loaded
flax_model_params = None
if pt_model_instance and pt_model_config:
    print("Converting PyTorch weights to Flax parameter structure...")
    flax_model_params = convert_pytorch_weights_to_flax(pt_model_instance.state_dict(), pt_model_config)
    print("Weight conversion complete.")
    # You can inspect the structure of flax_model_params here if needed:
    # print(jax.tree_map(lambda x: x.shape, flax_model_params))
else:
    print("Skipping weight conversion as PyTorch model was not loaded.")


# %% 5. Initializing the Flax Model with Ported Weights
print("\n--- 5. Initializing the Flax Model with Ported Weights ---\n")

flax_model_instance = None
if flax_model_params and pt_model_config:
    flax_model_instance = FlaxSmallGPT(
        vocab_size=pt_model_config['vocab_size'],
        embed_dim=pt_model_config['embed_dim'],
        num_heads=pt_model_config['num_heads'],
        num_layers=pt_model_config['num_layers'],
        block_size=pt_model_config['block_size'],
        dropout_rate=pt_model_config['dropout_rate'],
        deterministic=True # Set deterministic=True for inference (disables dropout)
    )
    print("Flax model instance created.")
    
    # To verify the parameter structure or if model.init is needed before apply:
    # dummy_input_ids_flax = jnp.ones((1, pt_model_config['block_size']), dtype=jnp.int32)
    # key_init = jax.random.PRNGKey(SEED) 
    # # If dropout layers are present, they might need their own RNG streams in 'dropout' etc.
    # # For deterministic=True, 'params' only should be fine.
    # initial_flax_params = flax_model_instance.init({'params': key_init, 'dropout': key_init}, dummy_input_ids_flax)
    # print("Flax model initialized with dummy data to get param structure (if needed).")
    # # Now, flax_model_params (converted from PyTorch) should match the structure of initial_flax_params['params']
else:
    print("Flax model instance not created as parameters were not converted.")


# %% 6. Implementing and Comparing Inference in PyTorch and JAX/Flax
print("\n--- 6. Implementing and Comparing Inference ---\n")

# --- Inference Function (PyTorch) ---
def generate_text_pytorch(model, tokenizer_pt, seed_text, max_len, block_size_pt, device_pt):
    model.eval()
    input_ids = tokenizer_pt.encode(seed_text)
    generated_ids_pt = list(input_ids)
    current_context_pt = torch.tensor([input_ids[-block_size_pt:]], dtype=torch.long, device=device_pt)

    with torch.no_grad():
        for _ in range(max_len):
            if current_context_pt.shape[1] > block_size_pt:
                current_context_pt = current_context_pt[:, -block_size_pt:]
            
            logits_pt = model(current_context_pt)
            next_token_logits_pt = logits_pt[:, -1, :]
            next_token_id_pt = torch.argmax(next_token_logits_pt, dim=-1).item()
            
            generated_ids_pt.append(next_token_id_pt)
            if next_token_id_pt == tokenizer_pt.eos_token_id: break
            
            next_token_tensor_pt = torch.tensor([[next_token_id_pt]], dtype=torch.long, device=device_pt)
            current_context_pt = torch.cat((current_context_pt, next_token_tensor_pt), dim=1)
            
    return tokenizer_pt.decode(generated_ids_pt)

# --- Inference Function (JAX/Flax) ---
# For JAX, the apply function needs to be JIT-compiled for performance.
# The generation loop itself runs in Python.
def flax_model_apply_for_inference(params, model_definition, input_ids_flax_jnp):
    # Dropout is disabled by `deterministic=True` in model definition
    # or by not passing a 'dropout' RNG key if model expects it.
    return model_definition.apply(params, input_ids_flax_jnp, rngs={'dropout': jax.random.key(0)}) # Dummy key for dropout if needed

# Apply jit after defining the function
flax_model_apply_for_inference_jit = jax.jit(flax_model_apply_for_inference, static_argnums=(1,))

def generate_text_flax(flax_params_tree, model_def_flax, tokenizer_flax, seed_text, max_len, block_size_flax):
    input_ids_flax = tokenizer_flax.encode(seed_text)
    generated_ids_flax = list(input_ids_flax)
    current_context_flax_np = np.array([input_ids_flax[-block_size_flax:]], dtype=np.int32)

    for _ in range(max_len):
        if current_context_flax_np.shape[1] > block_size_flax:
            current_context_flax_np = current_context_flax_np[:, -block_size_flax:]
        
        current_context_flax_jnp = jnp.asarray(current_context_flax_np)
        
        #logits_flax = model_def_flax.apply(flax_params_tree, current_context_flax_jnp)
        # For JITted version:
        logits_flax = flax_model_apply_for_inference_jit(flax_params_tree, model_def_flax, current_context_flax_jnp)

        next_token_logits_flax = logits_flax[:, -1, :]
        next_token_id_flax = jnp.argmax(next_token_logits_flax, axis=-1).item()
        
        generated_ids_flax.append(next_token_id_flax)
        if next_token_id_flax == tokenizer_flax.eos_token_id: 
            break
            
        current_context_flax_np = np.concatenate(
            (current_context_flax_np, np.array([[next_token_id_flax]], dtype=np.int32)), axis=1
        )
            
    return tokenizer_flax.decode(generated_ids_flax)


# --- Run Comparison ---
seed_prompt = "Moritz Laurer, born in Augsburg"

if pt_model_instance and flax_model_instance and flax_model_params:
    gen_max_len = 30 # Keep short for quick test
    
    print(f"\nGenerating with PyTorch model (on {pt_device}):")
    pt_generated_text = generate_text_pytorch(
        pt_model_instance, tokenizer, seed_prompt, gen_max_len, 
        pt_model_config['block_size'], pt_device
    )
    print(f"  PT Output: {pt_generated_text}")

    print(f"\nGenerating with JAX/Flax model (on {jax_device}):")
    # Ensure flax_model_params are on the correct JAX device (usually handled by JAX implicitly or device_put)
    # flax_model_params_on_device = jax.device_put(flax_model_params, jax_device) # If needed
    
    flax_generated_text = generate_text_flax(
        flax_model_params, flax_model_instance, tokenizer, seed_prompt, gen_max_len,
        pt_model_config['block_size'] # Use same block_size
    )
    print(f"  JAX Output: {flax_generated_text}")

    if pt_generated_text == flax_generated_text:
        print("\nSUCCESS: PyTorch and JAX/Flax outputs match!")
    else:
        print("\nMISMATCH: PyTorch and JAX/Flax outputs differ. Debugging needed.")
        print("Potential reasons: incorrect weight mapping/transposition, dropout state, mask differences, numerical precision.")
else:
    print("\nSkipping inference comparison as models/params were not fully set up.")


# %% Conclusion
print("\n--- Module 13 Summary ---\n")
print("Key Takeaways:")
print("- Translating models involves defining an equivalent architecture in the target framework (Flax).")
print("- Weight mapping is crucial and often complex, requiring careful attention to layer types,")
print("  parameter names (e.g., 'weight' vs 'kernel'), and tensor shapes (transpositions).")
print("- PyTorch's `nn.MultiheadAttention` `in_proj_weight` needs to be split and transposed for Flax `Dense` layers.")
print("- Flax models are stateless; parameters are passed explicitly to the `apply` method.")
print("- JIT compilation (`@jax.jit`) is standard for performant JAX inference.")
print("- Verification by comparing outputs on the same input is essential to confirm successful translation.")
print("\nThis process, while detailed, enables leveraging models across different deep learning ecosystems.")

print("\nEnd of Module 13.")