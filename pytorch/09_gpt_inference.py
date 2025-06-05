# -*- coding: utf-8 -*-
# 09_gpt_inference.py

# Module 9: Generating Text with the Trained Model
#
# This script demonstrates how to:
# 1. Load a pre-trained GPT-like model (saved from Module 8).
# 2. Understand and use inference mode settings: `model.eval()` and `torch.no_grad()`.
# 3. Implement autoregressive text generation.
# 4. Explore different decoding strategies:
#    - Greedy search
#    - Sampling with temperature
#    - Top-k filtering
# 5. Generate text examples.
# 6. Briefly compare with JAX-based generation.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import os
import random
import numpy as np

print("--- Module 9: Generating Text with the Trained Model ---\n")
print(f"Using PyTorch version: {torch.__version__}")

# --- Reproducibility ---
# Set seeds for reproducibility if any random operations are part of this script
# (though generation itself can be stochastic)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
print(f"Set random, numpy, and torch seeds to: {SEED}")

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

# %% 1. Configuration and Model/Tokenizer Setup
print("\n--- 1. Configuration and Model/Tokenizer Setup ---\n")

# --- Specify Checkpoint Path ---
# IMPORTANT: Update this path to point to the checkpoint saved by 08_full_gpt_model_train.py
# You can list files in './checkpoints' to find the exact name.
CHECKPOINT_PATH = './checkpoints/run_20250515_105608_E1_S20000_D128H8L6/checkpoint_step_54000.pth' # <--- USER NEEDS TO VERIFY THIS

if not os.path.exists(CHECKPOINT_PATH):
    print(f"WARNING: Checkpoint path '{CHECKPOINT_PATH}' does not exist.")
    print("Please ensure 08_full_gpt_model_train.py has been run and saved a checkpoint,")
    print("or update CHECKPOINT_PATH to the correct file.")
    # We might proceed with uninitialized model for structure demo, but generation will be random.
    # For this module, we'll try to load, and if it fails, model will be None.

# --- Load GPT-2 Tokenizer ---
# This should be the same tokenizer used during training.
print("Loading GPT-2 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# MODEL_CONFIG from checkpoint will have vocab_size, but good to have tokenizer instance.
print(f"GPT-2 Tokenizer loaded. Vocab size from tokenizer: {tokenizer.vocab_size}")


# %% 2. Model Definitions (Copied from Module 8 for self-containment)
print("\n--- 2. Model Definitions ---\n")

class TransformerBlock(nn.Module):
    """A single Transformer block for a decoder-style GPT."""
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
            nn.Dropout(dropout_rate)
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        _batch_size, seq_len, _embed_dim = x.shape
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            sz=seq_len,
            device=x.device,
            dtype=x.dtype # Ensures mask dtype matches input tensor x
        )
        
        norm_x = self.ln1(x)
        attn_output, _ = self.attn(norm_x, norm_x, norm_x, 
                                   attn_mask=causal_mask, 
                                   is_causal=False, 
                                   need_weights=False)
        x = x + self.dropout(attn_output) # Residual connection for attention

        norm_x = self.ln2(x)
        ff_output = self.ffn(norm_x)
        x = x + ff_output # Residual connection for FFN
        return x

class SmallGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, block_size, dropout_rate):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(block_size, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, dropout_rate=dropout_rate) for _ in range(num_layers)]
        )
        self.ln_final = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.token_embedding.weight = self.lm_head.weight # Weight tying (if used in training)

    def forward(self, idx_sequences):
        batch_size, seq_len = idx_sequences.shape
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

print("Model classes (TransformerBlock, SmallGPT) defined.")


# %% 3. Loading the Trained Model
print("\n--- 3. Loading the Trained Model ---\n")

def load_model_for_inference(model_class, checkpoint_path, device_to_load_on):
    """Loads a model from a checkpoint for inference."""
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint path '{checkpoint_path}' not found. Cannot load model.")
        return None, None

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device_to_load_on)
    
    model_config = checkpoint.get('model_config')
    if not model_config:
        print("Error: 'model_config' not found in checkpoint. Cannot instantiate model.")
        return None, None

    # Ensure vocab_size from tokenizer matches if it's in model_config
    if tokenizer.vocab_size != model_config.get('vocab_size'):
        print(f"Warning: Tokenizer vocab size ({tokenizer.vocab_size}) "
              f"differs from checkpoint model_config vocab_size ({model_config.get('vocab_size')}). "
              f"Using vocab_size from model_config for model instantiation.")
    
    # Use vocab_size from the checkpoint's model_config
    loaded_vocab_size = model_config['vocab_size']

    model = model_class(
        vocab_size=loaded_vocab_size,
        embed_dim=model_config['embed_dim'],
        num_heads=model_config['num_heads'],
        num_layers=model_config['num_layers'],
        block_size=model_config['block_size'],
        dropout_rate=model_config['dropout_rate']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device_to_load_on)
    
    print(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'N/A')}.")
    print(f"Loaded Model Config: {model_config}")
    return model, model_config


# Attempt to load the model
model, loaded_model_config = load_model_for_inference(SmallGPT, CHECKPOINT_PATH, device)


# Optional: Compile the model for potentially faster inference
# Note: Compilation can take time.
compile_model = False
if compile_model:
    print("Compiling the loaded model with torch.compile()...")
    try:
        model = torch.compile(model)
        print("Model compiled successfully for inference.")
    except Exception as e:
        print(f"Failed to compile model: {e}. Proceeding with uncompiled model.")
else:
    print("Skipping model compilation with torch.compile().")

# This block_size will be used in generation to manage context.
# It should come from the loaded model's configuration.
BLOCK_SIZE = loaded_model_config['block_size'] if loaded_model_config else 64


# %% 4. Inference Mode: `model.eval()` and `torch.no_grad()`
print("\n--- 4. Inference Mode: `model.eval()` and `torch.no_grad()` ---\n")

print("During inference (text generation), we need to configure PyTorch appropriately:")
print("1. `model.eval()`: This sets the model to evaluation mode.")
print("   - Disables layers like Dropout (they behave as pass-through).")
print("   - Makes Batch Normalization layers use their learned running statistics instead of batch statistics.")
print("   - Crucial for consistent and reproducible outputs.")
print("   (Our SmallGPT uses Dropout, so model.eval() is important.)")

print("\n2. `with torch.no_grad():`: This context manager disables gradient calculations.")
print("   - Reduces memory consumption because PyTorch doesn't need to store intermediate values for backpropagation.")
print("   - Speeds up computations as gradient tracking overhead is removed.")
print("   - Essential for inference as we are not training the model.")

if model:
    model.eval() # Set the loaded model to evaluation mode
    print("\nModel set to evaluation mode (`model.eval()`).")


# %% 5. Autoregressive Text Generation
print("\n--- 5. Autoregressive Text Generation ---\n")

print("Autoregressive generation means producing text one token at a time:")
print("1. Start with a seed (prompt) sequence of tokens.")
print("2. Feed the current sequence to the model to get logits (predictions) for the next token.")
print("3. Convert logits to probabilities (usually via softmax).")
print("4. Select the next token based on these probabilities (e.g., greedy, sampling).")
print("5. Append the new token to the sequence.")
print("6. Repeat from step 2 with the updated sequence until a desired length or an End-Of-Sequence (EOS) token is generated.")
print("The model's context window (`block_size`) means only the last `block_size` tokens are typically used as input for the next prediction.")


# %% 6. Decoding Strategies and Implementation
print("\n--- 6. Decoding Strategies and Implementation ---\n")

def generate_text(
    model_instance, 
    start_string, 
    tokenizer_instance, 
    max_len=50, 
    block_size_from_config=64,
    temperature=1.0, 
    top_k=None,
    device_for_generation="cpu"
):
    """
    Generates text autoregressively using the given model and decoding strategy.

    Args:
        model_instance (nn.Module): The trained GPT model.
        start_string (str): The initial text prompt.
        tokenizer_instance: The tokenizer used for training.
        max_len (int): Maximum number of new tokens to generate.
        block_size_from_config (int): The context window size of the model.
        temperature (float): Controls randomness. Higher values (e.g., >1) make output more random,
                             lower values (e.g., <1) make it more deterministic. 0 means greedy.
        top_k (int, optional): If set, filters to the k most likely next tokens before sampling.
        device_for_generation (str): Device to run generation on ('cpu', 'cuda', 'mps').
    
    Returns:
        str: The generated text sequence.
    """
    if not model_instance:
        return "Model not loaded. Cannot generate text."

    model_instance.eval() # Ensure model is in eval mode
    
    print(f"\nGenerating text (max_len={max_len}, temp={temperature}, top_k={top_k}) from seed: '{start_string}'")
    
    input_ids = tokenizer_instance.encode(start_string)
    generated_ids = list(input_ids) # Full sequence of generated token IDs

    # Ensure current_context_ids is a 2D tensor (batch_size=1, seq_len)
    current_context_ids = torch.tensor([input_ids[-block_size_from_config:]], dtype=torch.long, device=device_for_generation)

    with torch.no_grad():
        for i in range(max_len):
            # Ensure context does not exceed block_size
            if current_context_ids.shape[1] > block_size_from_config:
                current_context_ids = current_context_ids[:, -block_size_from_config:]

            logits = model_instance(current_context_ids) # (1, current_seq_len, vocab_size)
            next_token_logits = logits[:, -1, :] # Focus on the last token: (1, vocab_size)

            # --- Greedy Search (if temperature is 0 or very low) ---
            if temperature == 0.0: # Pure greedy
                next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            else:
                # --- Apply Temperature Scaling ---
                next_token_logits = next_token_logits / temperature

                # --- Apply Top-k Filtering (Optional) ---
                if top_k is not None and top_k > 0:
                    # Remove tokens with probability less than the Kth token
                    # Sort logits, get the Kth value
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    # Set logits for tokens not in top-k to -infinity
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                
                # --- Convert Logits to Probabilities ---
                probs = F.softmax(next_token_logits, dim=-1) # (1, vocab_size)
                
                # --- Sample the Next Token ---
                next_token_id = torch.multinomial(probs, num_samples=1).item() # Get scalar Python number
            
            generated_ids.append(next_token_id)
            
            # Update context for the next iteration
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device_for_generation)
            current_context_ids = torch.cat((current_context_ids, next_token_tensor), dim=1)
            
            # Stop if EOS token is generated
            if tokenizer_instance.eos_token_id is not None and next_token_id == tokenizer_instance.eos_token_id:
                print(f"EOS token ({tokenizer_instance.eos_token_id}) generated at step {i+1}. Stopping.")
                break
            
    return tokenizer_instance.decode(generated_ids)


# --- Example Usage of Text Generation ---
if model and loaded_model_config:
    print("\n--- Generating Text Examples ---")
    
    # Example 1: More deterministic (lower temperature)
    seed_text_1 = "The future of AI is"
    generated_sequence_1 = generate_text(
        model, seed_text_1, tokenizer, 
        max_len=70, block_size_from_config=BLOCK_SIZE, 
        temperature=0.7, top_k=40, device_for_generation=device
    )
    print("\nGenerated Text 1 (temp=0.7, top_k=40):")
    print(generated_sequence_1)

    # Example 2: More creative/random (higher temperature)
    seed_text_2 = "Once upon a time, in a land far away,"
    generated_sequence_2 = generate_text(
        model, seed_text_2, tokenizer,
        max_len=70, block_size_from_config=BLOCK_SIZE,
        temperature=1.0, top_k=50, device_for_generation=device
    )
    print("\nGenerated Text 2 (temp=1.0, top_k=50):")
    print(generated_sequence_2)

    # Example 3: Greedy (temperature=0)
    seed_text_3 = "PyTorch is a library for"
    generated_sequence_3 = generate_text(
        model, seed_text_3, tokenizer,
        max_len=50, block_size_from_config=BLOCK_SIZE,
        temperature=0.0, device_for_generation=device # top_k is irrelevant for greedy
    )
    print("\nGenerated Text 3 (greedy, temp=0.0):")
    print(generated_sequence_3)
else:
    print("\nSkipping text generation examples as model was not loaded successfully.")


# %% 7. JAX Comparison for Text Generation
print("\n--- 7. JAX Comparison for Text Generation ---\n")

print("Text generation in JAX (e.g., with Flax or Haiku) has some differences:")
print("- Explicit State & PRNG Keys: JAX is functional. Model parameters and any state (like for RNNs, though less common in pure Transformers for generation state beyond keys)")
print("  are explicitly passed into and out of functions. Stochastic operations like sampling require explicit PRNG keys, which must be managed (split for each step).")
print("- JIT Compilation: The entire generation step (forward pass, sampling) is often JIT-compiled (`@jax.jit`) for performance.")
print("  This means Python loops for autoregressive generation run outside the JIT-compiled step function, calling it repeatedly.")
print("- Loop Structure: A Python `for` loop typically manages the autoregressive token-by-token generation,")
print("  similar to PyTorch, but it calls the JITted JAX function for each step.")
print("- Optimization: Compiling the generation step can lead to very efficient execution, especially on TPUs.")
print("- Frameworks: Libraries like Flax provide utilities, but the core loop and PRNG key handling are fundamental JAX concepts.")


# %% Conclusion
print("\n--- Module 9 Summary ---\n")
print("Key Takeaways:")
print("- Loading trained models involves restoring `state_dict` and model configuration.")
print("- `model.eval()` and `torch.no_grad()` are crucial for correct and efficient inference.")
print("- Autoregressive generation builds text token by token, using the model's output as input for the next step.")
print("- Decoding strategies like greedy search, temperature sampling, and top-k filtering control the trade-off between coherence and creativity.")
print("- PyTorch provides flexible tools to implement these generation techniques.")

print("\nEnd of Module 9.")