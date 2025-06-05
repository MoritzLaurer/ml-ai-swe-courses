# -*- coding: utf-8 -*-
# 18_sota_gpt_train.py

# Module 18: SOTA GPT Model Training with Advanced Architectural Components
#
# This script builds upon 08_full_gpt_model_train.py by incorporating several
# state-of-the-art (SOTA) architectural features found in modern LLMs. 
# The goal is to demonstrate their implementation within a small-scale model.
#
# This script demonstrates:
# 1. Defining components of a GPT-like model using SOTA architectural components:
#    - RoPE positional embeddings
#    - Grouped Query Attention (GQA)
#    - Alternating Global and Local Sliding Window Attention
#    - RMSNorm
#    - SwiGLU Feed-Forward Network (FFN)
# 2. Assembling these components into a full nn.Module.
# 3. Preparing a character-level dataset and dataloader for training.
# 4. Implementing a training loop for the language model.
# 5. A simple example of generating text after training.

# Architectural Enhancements Compared to 08_full_gpt_model_train.py:
#
# 1. RMSNorm (Root Mean Square Normalization):
#    - What: A simpler normalization technique than LayerNorm.
#    - Why: Reduces computational overhead and memory usage, often with comparable or 
#           better performance. Standard in models like Llama, Gemma.
#
# 2. RoPE (Rotary Positional Embeddings):
#    - What: Encodes positional information by rotating query and key embeddings.
#    - Why: Offers better generalization to different sequence lengths and has become 
#           a standard in many SOTA models (Llama, Qwen, Gemma).
#
# 3. SwiGLU Feed-Forward Network (FFN):
#    - What: A type of Gated Linear Unit using Swish activation.
#    - Why: Often allows for a reduction in FFN hidden dimension while maintaining 
#           or improving performance. Used in Llama, Qwen.
#
# 4. Grouped Query Attention (GQA):
#    - What: Allows multiple query heads to share key/value heads.
#    - Why: Reduces KV cache size and computational load during inference, leading 
#           to faster inference and lower memory. Used in Llama, Gemma, Qwen.
#
# 5. Alternating Local and Global Attention:
#    - What: Most layers use local (sliding window) attention, with occasional 
#           global attention layers.
#    - Why: Significantly reduces computational cost and memory for long sequences, 
#           making longer context windows feasible. Seen in Gemma, Command-A.
#
# Note on Training Speed for this Script:
# While these architectural components are designed for SOTA performance and efficiency 
# at scale, their introduction in this small-scale model, without `torch.compile()` 
# and with a small `block_size`, may not result in faster training speeds compared to 
# the simpler architecture in `08_full_gpt_model_train.py`. This is due to:
#   - Increased number of operations in some custom modules (e.g., SwiGLU, RoPE application).
#   - Overhead of Python logic for custom components and mask generation.
#   - Potentially less optimized kernel usage compared to built-in modules like `nn.MultiheadAttention` 
#     (used in the original script) which benefit from fused operations and C++/CUDA/MPS implementations.
#   - The relatively small `block_size` where the asymptotic benefits of techniques like 
#     local attention are less pronounced.
# The primary purpose here is educational: to demonstrate implementation.
#
# Original Script Features Retained:
# - Basic GPT-like model structure (embeddings, transformer blocks, LM head).
# - GPT-2 tokenizer and FineWeb-Edu dataset sample.
# - Training loop, checkpointing, and simple text generation.



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import random # Added for setting seed
import numpy as np # Added for setting seed
from transformers import AutoTokenizer # Added for GPT-2 tokenizer
from tqdm import tqdm # <--- Add this import
import os # Add this for path manipulation
import time
import math # Add math for sqrt in weight initialization

print("--- Module 18: SOTA GPT Model Training ---\
")
print(f"Using PyTorch version: {torch.__version__}")

# --- Reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# Potentially add for cudnn determinism, though can impact performance
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
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


# %% 1. Configuration Parameters
print("\n--- 1. Configuration Parameters ---\n")

# Model Hyperparameters
MODEL_CONFIG = {
    'vocab_size': None,      # Will be set from tokenizer
    'embed_dim': 128,        # Embedding dimension
    'num_q_heads': 8,        # Number of query attention heads (num_heads -> num_q_heads)
    'num_kv_heads': 2,       # Number of key/value attention heads for GQA (must ensure num_q_heads % num_kv_heads == 0)
    'num_layers': 6,         # Number of Transformer blocks
    'block_size': 128,       # Maximum sequence length (context window)
    'dropout_rate': 0.1,     # Dropout rate
    'ff_hidden_dim_factor': 8/3, # Hidden dimension factor for SwiGLU FFN (e.g., 2/3 * 4 = 8/3 for Llama-style)
    'rms_norm_eps': 1e-5,    # Epsilon for RMSNorm (Llama uses 1e-5)
    'rope_base': 10000.0,    # RoPE base frequency
    'local_attn_window_size': 32, # Window size for local attention (e.g., 128, 256, 512). Set <=0 to disable local attention.
    'global_every_n_layers': 3   # Every Nth layer is global. E.g., 3 means layers 2, 5, 8... (0-indexed) are global.
                                 # If 0 or 1, all layers are global. If > num_layers, all effectively local (except first if 0-indexed logic applies based on %).
}

# Training Hyperparameters
TRAIN_CONFIG = {
    'batch_size': 128,
    'learning_rate': 3e-4,
    'num_epochs': 1, # Keep small for quick demo
    'data_seed': 1234, # Seed specific for dataset shuffling
    'num_text_samples_to_load': 20000,
    'load_specific_checkpoint_path': None, # Set this to a path string to load a specific checkpoint
                                           # e.g., './checkpoints/gpt_model_checkpoint.pth'
                                           # If None, trains from scratch.
    'checkpoint_base_dir': './checkpoints', # Base directory for all runs
    'save_every_n_steps': 2000, # Save a checkpoint every N global steps. Set to 0 or less to disable step-based saving.
    'max_checkpoints_to_keep': 3,  # Keep the latest N checkpoints. Set to 0 or less to keep all.
}

# Generate unique directory name for this specific run's checkpoints
run_specific_dir_name = f'run_{time.strftime("%Y%m%d_%H%M%S")}_E{TRAIN_CONFIG["num_epochs"]}_S{TRAIN_CONFIG["num_text_samples_to_load"]}_D{MODEL_CONFIG["embed_dim"]}Q{MODEL_CONFIG["num_q_heads"]}KV{MODEL_CONFIG["num_kv_heads"]}L{MODEL_CONFIG["num_layers"]}'
TRAIN_CONFIG['full_run_checkpoint_dir'] = os.path.join(TRAIN_CONFIG['checkpoint_base_dir'], run_specific_dir_name)

# Create the run-specific checkpoint directory if it doesn't exist
if not os.path.exists(TRAIN_CONFIG['full_run_checkpoint_dir']):
    os.makedirs(TRAIN_CONFIG['full_run_checkpoint_dir'])
    print(f"Created run-specific checkpoint directory: {TRAIN_CONFIG['full_run_checkpoint_dir']}")


# Update VOCAB_SIZE in MODEL_CONFIG after tokenizer is loaded
# (This will happen in Section 2)

print(f"Model Config: {MODEL_CONFIG}")
print(f"Training Config: {TRAIN_CONFIG}")
print(f"Checkpoints for this run will be saved in: {TRAIN_CONFIG['full_run_checkpoint_dir']}")


# %% 2. Data Preparation (GPT-2 Tokenizer)
print("\n--- 2. Data Preparation (GPT-2 Tokenizer) ---\n")

# --- Load GPT-2 Tokenizer ---
print("Loading GPT-2 tokenizer...")
# Using a standard GPT-2 tokenizer.
# You can replace "gpt2" with other models like "gpt2-medium", etc.
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# GPT-2 tokenizer doesn't have a PAD token by default.
# If we were to use padding for sequences of different lengths directly in a batch
# (e.g. if not using fixed-size chunks from the dataset), we might set one.
# For this script, fixed-length chunks are used, so explicit padding at batch time isn't strictly needed.
# Example: tokenizer.pad_token = tokenizer.eos_token

MODEL_CONFIG['vocab_size'] = tokenizer.vocab_size # Set vocab size in config
print(f"GPT-2 Tokenizer loaded. Vocab size: {MODEL_CONFIG['vocab_size']}")
if tokenizer.eos_token:
    print(f"EOS token: '{tokenizer.eos_token}', ID: {tokenizer.eos_token_id}")
if tokenizer.bos_token:
    print(f"BOS token: '{tokenizer.bos_token}', ID: {tokenizer.bos_token_id}")


# --- Load a sample from FineWeb-Edu ---
print("Loading a small sample from HuggingFaceFW/fineweb-edu...")
# Stream a small number of examples to avoid downloading the whole dataset
# The 'text' field contains the document content
# We take N examples and join them to form a single corpus string
# Buffer size for shuffling: needs to be large enough for a decent shuffle
# but small enough not to consume too much memory.
# For taking N samples, a buffer somewhat larger than N is good.
shuffle_buffer_size = TRAIN_CONFIG['num_text_samples_to_load'] * 5 # e.g., 500 for 100 samples

fw_edu_sample_stream = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)

# Shuffle the stream with a seed before taking samples for reproducibility
# This ensures we get the same "random" 100 samples each time.
print(f"Shuffling dataset stream with seed {TRAIN_CONFIG['data_seed']} and buffer size {shuffle_buffer_size} before taking samples.")
shuffled_stream = fw_edu_sample_stream.shuffle(seed=TRAIN_CONFIG['data_seed'], buffer_size=shuffle_buffer_size)

corpus_parts = []
# Iterate over the shuffled stream and take the desired number of samples
for i, example in enumerate(shuffled_stream.take(TRAIN_CONFIG['num_text_samples_to_load'])):
    if 'text' in example and example['text']: # Ensure text field exists and is not empty
        corpus_parts.append(example['text'])
        
if not corpus_parts:
    raise ValueError("No text data loaded from the dataset sample. Check dataset structure or increase sample size.")

print(f"Successfully loaded {len(corpus_parts)} documents from FineWeb-Edu sample.")

# --- Tokenize and Concatenate Sampled Documents ---
print("Tokenizing and concatenating sampled documents with EOS separator...")
print("Corpus is one long string with EOS tokens as separators between documents.")
encoded_corpus_ids = []
for i, text_doc in enumerate(corpus_parts):
    # Tokenize each document. Using strip() here just in case.
    # tokenizer.encode does not add special tokens by default.
    doc_ids = tokenizer.encode(text_doc.strip()) 
    encoded_corpus_ids.extend(doc_ids)
    
    # Add EOS token as a separator between documents, but not after the last one.
    if tokenizer.eos_token_id is not None and i < len(corpus_parts) - 1:
        encoded_corpus_ids.append(tokenizer.eos_token_id)

print(f"Sampled documents tokenized and concatenated. Total GPT-2 tokens: {len(encoded_corpus_ids)}")

# --- Custom Dataset for Language Modeling (using GPT-2 tokens) ---
class GPT2TokenLMDataset(Dataset):
    """Custom PyTorch Dataset for preparing (input, target) token sequences for language modeling."""
    def __init__(self, token_ids_list, block_size):
        self.token_ids = token_ids_list # Should be a flat list of token IDs from the entire corpus
        self.block_size = block_size
        
        # Calculate the number of possible sequences.
        # Each sequence requires (block_size + 1) tokens to form one (x, y) pair.
        num_possible_sequences = len(self.token_ids) - self.block_size
        
        if num_possible_sequences <= 0:
            raise ValueError(
                f"Tokenized corpus is too short for the given block_size. "
                f"Need at least {self.block_size + 1} tokens, but got {len(self.token_ids)}."
            )
        print(f"Dataset created with {num_possible_sequences} possible token sequences of length {block_size}.")

    def __len__(self):
        # Number of possible starting points for a sequence of (block_size + 1) tokens
        return len(self.token_ids) - self.block_size

    def __getitem__(self, idx):
        # Input sequence (context)
        chunk_x = self.token_ids[idx : idx + self.block_size]
        # Target sequence (next token prediction)
        chunk_y = self.token_ids[idx + 1 : idx + self.block_size + 1]
        
        return torch.tensor(chunk_x, dtype=torch.long), \
               torch.tensor(chunk_y, dtype=torch.long)

# Instantiate the new dataset and dataloader
# The GPT2TokenLMDataset will raise a ValueError if encoded_corpus_ids is too short.
train_dataset = GPT2TokenLMDataset(encoded_corpus_ids, MODEL_CONFIG['block_size'])
train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True, generator=torch.Generator().manual_seed(SEED)) # Add generator for reproducible shuffling

# Test dataset and loader (using the tokenizer's decode method)
sample_x_ids, sample_y_ids = train_dataset[0]
print(f"Sample input token IDs: {sample_x_ids.shape}\n{sample_x_ids[:10]}...") # Print first 10
print(f"Decoded input (first 50 chars): {tokenizer.decode(sample_x_ids.tolist())[:50]}...")
print(f"Sample target token IDs: {sample_y_ids.shape}\n{sample_y_ids[:10]}...") # Print first 10
print(f"Decoded target (first 50 chars): {tokenizer.decode(sample_y_ids.tolist())[:50]}...")


# %% 3. SOTA Architectural Components
print("\n--- 3. SOTA Architectural Components ---\
")

# Root Mean Square Layer Normalization.
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# Cache for RoPE (Rotary Positional Embedding) frequencies.
class RoPECache(nn.Module):
    """Cache for RoPE frequencies."""
    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0, device=None):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.base = base
        self.dim = dim

        # Generates theta_i = 1 / (base^(2i / dim))
        # Shape: (dim / 2)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute frequencies for all positions up to max_seq_len
        self._build_cache()

    def _build_cache(self):
        t = torch.arange(self.max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        # freqs shape: (max_seq_len, dim / 2)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # freqs_cis shape: (max_seq_len, dim / 2 * 2) -> (max_seq_len, dim)
        # Converts to complex numbers: cos(m*theta) + i*sin(m*theta)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, seq_len: int):
        # Return precomputed_freqs_cis for the given sequence length
        # freqs_cis shape: (seq_len, dim)
        return self.freqs_cis[:seq_len]


# Applies RoPE (Rotary Positional Embedding) to query and key tensors.
def apply_rotary_pos_emb(q, k, freqs_cis):
    """Applies RoPE to query and key tensors."""
    # q, k: (batch, seq_len, num_heads, head_dim)
    # freqs_cis: (seq_len, head_dim)

    # Reshape q, k to (batch, seq_len, num_heads, head_dim // 2, 2) to view as complex numbers
    # Then convert to complex: (batch, seq_len, num_heads, head_dim // 2)
    q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))

    # Reshape freqs_cis to be broadcastable: (1, seq_len, 1, head_dim // 2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2) # 1, seq_len, 1, head_dim/2

    # Apply rotation in complex plane: (query_rotated)_i = query_i * exp(i * m * theta_i)
    # q_complex * freqs_cis : (batch, seq_len, num_heads, head_dim//2)
    q_rotated = torch.view_as_real(q_complex * freqs_cis).flatten(3).type_as(q)
    k_rotated = torch.view_as_real(k_complex * freqs_cis).flatten(3).type_as(k)

    return q_rotated, k_rotated


# SwiGLU (Swish Gated Linear Unit) Feed-Forward Network.
class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network."""
    def __init__(self, dim: int, hidden_dim_factor: float, dropout_rate: float = 0.1):
        super().__init__()
        hidden_dim = int(hidden_dim_factor * dim) # Llama uses 2/3 factor on hidden_dim, so 2/3 * 4 = 8/3 of dim
        # Ensure hidden_dim is a multiple of a preferred value (e.g., 256, from Llama)
        # This helps with hardware efficiency. For smaller models, this might be less critical
        # but good practice. Here, we'll just use the calculated hidden_dim.

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False) # For the gating mechanism
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: (batch, seq_len, dim)
        # F.silu(w1(x)) * w3(x)
        swish_gate = F.silu(self.w1(x)) * self.w3(x)
        output = self.w2(swish_gate)
        return self.dropout(output)


# Grouped Query Attention (GQA) module with RoPE.
class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA) with RoPE."""
    def __init__(self, embed_dim: int, num_q_heads: int, num_kv_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
        
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_q_heads
        self.num_q_per_kv = num_q_heads // num_kv_heads # Number of Q heads per KV group

        self.Wq = nn.Linear(embed_dim, num_q_heads * self.head_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.Wo = nn.Linear(num_q_heads * self.head_dim, embed_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout_rate)
        self.resid_dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor = None):
        # x shape: (batch_size, seq_len, embed_dim)
        # freqs_cis shape: (seq_len, head_dim)
        # mask shape: (seq_len, seq_len)
        batch_size, seq_len, _ = x.shape

        # 1. Linear projections for Q, K, V
        # q: (batch, seq_len, num_q_heads * head_dim)
        # k, v: (batch, seq_len, num_kv_heads * head_dim)
        xq = self.Wq(x)
        xk = self.Wk(x)
        xv = self.Wv(x)

        # 2. Reshape Q, K, V for multi-head attention
        # q: (batch, seq_len, num_q_heads, head_dim)
        # k, v: (batch, seq_len, num_kv_heads, head_dim)
        xq = xq.view(batch_size, seq_len, self.num_q_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # 3. Apply RoPE embeddings to Q and K before repeating K/V
        # RoPE is applied per head, so head_dim should match freqs_cis dim
        xq, xk = apply_rotary_pos_emb(xq, xk, freqs_cis=freqs_cis)

        # 4. Grouped Query Attention: Repeat K and V heads to match Q heads
        # K_rep: (batch, seq_len, num_kv_heads, num_q_per_kv, head_dim)
        # K_rep: (batch, seq_len, num_q_heads, head_dim)
        if self.num_kv_heads != self.num_q_heads:
            xk = xk.unsqueeze(3).repeat(1, 1, 1, self.num_q_per_kv, 1).view(batch_size, seq_len, self.num_q_heads, self.head_dim)
            xv = xv.unsqueeze(3).repeat(1, 1, 1, self.num_q_per_kv, 1).view(batch_size, seq_len, self.num_q_heads, self.head_dim)
        
        # 5. Transpose for attention calculation: (batch, num_q_heads, seq_len, head_dim)
        q_trans = xq.transpose(1, 2)
        k_trans = xk.transpose(1, 2)
        v_trans = xv.transpose(1, 2)

        # 6. Scaled Dot-Product Attention
        # (batch, num_q_heads, seq_len, head_dim) @ (batch, num_q_heads, head_dim, seq_len) -> (batch, num_q_heads, seq_len, seq_len)
        attn_scores = torch.matmul(q_trans, k_trans.transpose(2, 3)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            # mask is (seq_len, seq_len), needs to be (1, 1, seq_len, seq_len) for broadcasting
            attn_scores = attn_scores + mask.unsqueeze(0).unsqueeze(0)
            
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # (batch, num_q_heads, seq_len, seq_len) @ (batch, num_q_heads, seq_len, head_dim) -> (batch, num_q_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_probs, v_trans)

        # 7. Concatenate heads and apply final linear layer
        # (batch, seq_len, num_q_heads, head_dim) -> (batch, seq_len, num_q_heads * head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.Wo(attn_output)
        return self.resid_dropout(output)


def generate_sliding_window_causal_mask(seq_len: int, window_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Generates a sliding window causal mask.
    Args:
        seq_len: The sequence length.
        window_size: The size of the attention window (number of previous tokens to attend to).
        device: The device to create the mask on.
        dtype: The data type for the mask.
    Returns:
        A (seq_len, seq_len) additive mask with -inf for masked positions and 0.0 for allowed positions.
    """
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
    # Create a band mask where each token can attend to `window_size` previous tokens and itself.
    # This is equivalent to a lower triangular matrix where elements outside the band are -inf.
    for i in range(seq_len):
        start_idx = max(0, i - window_size + 1) # +1 because the window includes the current token
        mask[i, start_idx : i + 1] = 0.0
    return mask


# %% 3. Transformer Block Component
print("\n--- 3. Transformer Block Component ---\
")

# A single Transformer block incorporating SOTA components (RMSNorm, GQA with RoPE, SwiGLU FFN).
class TransformerBlock(nn.Module):
    """A single Transformer block with SOTA components."""
    def __init__(self, embed_dim: int, num_q_heads: int, num_kv_heads: int, 
                 ff_hidden_dim_factor: float, rms_norm_eps: float, dropout_rate: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.attn = GroupedQueryAttention(embed_dim, num_q_heads, num_kv_heads, dropout_rate=dropout_rate)
        self.norm2 = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.ffn = SwiGLUFFN(embed_dim, ff_hidden_dim_factor, dropout_rate=dropout_rate)
        # Dropout is already applied within GQA and SwiGLUFFN, so a final dropout here might be redundant,
        # but can be kept for consistency if desired or if those modules didn't have internal dropout.
        # self.dropout = nn.Dropout(dropout_rate) # Consider if this is needed.

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor):
        # x shape: (batch_size, seq_len, embed_dim)
        # freqs_cis shape: (seq_len, head_dim_of_attention)
        # mask shape: (seq_len, seq_len)

        # Attention part with pre-normalization (RMSNorm)
        # The residual connection comes *after* the attention block.
        h = x + self.attn(self.norm1(x), freqs_cis=freqs_cis, mask=mask)

        # Feed-forward part with pre-normalization (RMSNorm)
        # The residual connection comes *after* the FFN block.
        out = h + self.ffn(self.norm2(h))
        return out

# Test the block
# test_block = TransformerBlock(MODEL_CONFIG['embed_dim'], MODEL_CONFIG['num_heads'], dropout_rate=MODEL_CONFIG['dropout_rate']).to(device)
# test_input_block = torch.randn(TRAIN_CONFIG['batch_size'], MODEL_CONFIG['block_size'], MODEL_CONFIG['embed_dim']).to(device)
# test_output_block = test_block(test_input_block)
# print(f"TransformerBlock test output shape: {test_output_block.shape}")


# %% 4. Full GPT-like Model Definition
print("\n--- 4. Full GPT-like Model Definition ---\n")

# The main GPT-like language model using SOTA architectural components.
class SmallGPT(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_q_heads: int, num_kv_heads: int, 
                 num_layers: int, block_size: int, dropout_rate: float, 
                 ff_hidden_dim_factor: float, rms_norm_eps: float, rope_base: float = 10000.0,
                 local_attn_window_size: int = -1, global_every_n_layers: int = 0): # Added new params with defaults
        super().__init__()
        self.block_size = block_size
        self.embed_dim = embed_dim
        self.num_q_heads = num_q_heads
        self.num_layers = num_layers
        self.head_dim = embed_dim // num_q_heads
        self.local_attn_window_size = local_attn_window_size
        self.global_every_n_layers = global_every_n_layers

        # Token embedding: maps character indices to vectors
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # RoPE Cache - no longer using learned positional embeddings
        self.rope_cache = RoPECache(dim=self.head_dim, max_seq_len=block_size, base=rope_base, device=device) 
        
        self.dropout = nn.Dropout(dropout_rate) # Dropout after embedding

        # Stack of Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                ff_hidden_dim_factor=ff_hidden_dim_factor,
                rms_norm_eps=rms_norm_eps,
                dropout_rate=dropout_rate
            ) for _ in range(num_layers)
        ])
        
        self.norm_final = RMSNorm(embed_dim, eps=rms_norm_eps) # Final RMSNorm
        # Language model head: linear layer to map to vocabulary logits
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False) # No bias, as per Llama

        # Weight tying (optional but common, can improve performance for smaller models)
        self.token_embedding.weight = self.lm_head.weight # Tie weights

        print("SmallGPT model initialized with SOTA components.")
        self._init_weights() # Apply custom weight initialization


    def _init_weights(self):
        # Initialize parameters (Llama-style)
        for name, param in self.named_parameters():
            if name.endswith("weight") and isinstance(param, nn.Linear):
                # Initialize linear layer weights with Kaiming normal for non-output layers
                # or Xavier uniform for output layers if preferred. Llama uses normal.
                nn.init.normal_(param, mean=0.0, std=0.02) # A common std for GPT-like models
            elif isinstance(param, nn.Embedding):
                nn.init.normal_(param.weight, mean=0.0, std=0.02)
            # Biases are mostly false in the new architecture, but if any exist:
            elif "bias" in name:
                nn.init.zeros_(param)
        
        # Apply special initialization for the output projection (Wo in GQA, w2 in SwiGLU)
        # This is important for stabilizing training with residual connections (GPT-2/LLaMA papers)
        for block in self.transformer_blocks:
            if hasattr(block.attn, 'Wo') and block.attn.Wo.weight is not None:
                # std=0.02 / math.sqrt(2 * self.num_layers) # GPT-2 style init for residual projections
                # Llama uses a different scheme, often tied to overall model depth or width.
                # Let's stick to a simpler std for now, or make it configurable if needed.
                nn.init.normal_(block.attn.Wo.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers))
            if hasattr(block.ffn, 'w2') and block.ffn.w2.weight is not None:
                nn.init.normal_(block.ffn.w2.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers))


    def forward(self, idx_sequences: torch.Tensor):
        # idx_sequences shape: (batch_size, seq_len) where seq_len <= block_size
        batch_size, seq_len = idx_sequences.shape

        # 1. Get token embeddings
        tok_emb = self.token_embedding(idx_sequences) # (batch, seq_len, embed_dim)
        x = self.dropout(tok_emb) # Apply dropout after embeddings
        
        # 2. Get RoPE frequencies for the current sequence length
        # freqs_cis shape: (seq_len, head_dim)
        freqs_cis = self.rope_cache(seq_len)
        
        # 3. Create attention masks
        global_causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=idx_sequences.device, dtype=x.dtype), diagonal=1)
        
        local_sliding_window_mask = None
        # Determine if local attention is generally applicable for this forward pass
        is_local_attention_configured = self.local_attn_window_size > 0 and self.local_attn_window_size < seq_len

        if is_local_attention_configured:
            local_sliding_window_mask = generate_sliding_window_causal_mask(
                seq_len, self.local_attn_window_size, device=idx_sequences.device, dtype=x.dtype
            )

        # 4. Pass through Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            current_mask = global_causal_mask # Default to global attention

            if is_local_attention_configured:
                # global_every_n_layers <= 1 means all layers are global (or effectively global if local not configured)
                if self.global_every_n_layers > 1: 
                    if (i + 1) % self.global_every_n_layers != 0: # This layer is local
                        if local_sliding_window_mask is not None:
                            current_mask = local_sliding_window_mask
                    # Else (it's a global layer in the alternating scheme), it uses global_causal_mask (already set)
                # Else (global_every_n_layers <= 1), all layers use global_causal_mask (already set)
            
            x = block(x, freqs_cis, current_mask)
        
        # 5. Final RMSNorm
        x = self.norm_final(x) # (batch, seq_len, embed_dim)
        
        # 6. Language Model Head
        logits = self.lm_head(x) # (batch, seq_len, vocab_size)
        
        return logits

# Instantiate the model
# We'll defer the main model instantiation to either loading or new creation
# model = SmallGPT(MODEL_CONFIG['vocab_size'], MODEL_CONFIG['embed_dim'], MODEL_CONFIG['num_heads'], 
#                  MODEL_CONFIG['num_layers'], MODEL_CONFIG['block_size'], MODEL_CONFIG['dropout_rate'])

# %% Helper functions for Saving and Loading Checkpoints
print("\n--- Helper functions for Saving and Loading Checkpoints ---\n")

# Saves model, optimizer, and training state to a checkpoint file.
def save_checkpoint(epoch, global_step, model, optimizer, model_config, train_config, loss):
    """Saves model, optimizer, and training state to the run-specific checkpoint directory."""
    run_checkpoint_dir = train_config['full_run_checkpoint_dir']
    
    # Ensure the run-specific directory exists (it should have been created earlier, but this is a safeguard)
    if not os.path.exists(run_checkpoint_dir):
        os.makedirs(run_checkpoint_dir)
        print(f"Created run-specific checkpoint directory during save: {run_checkpoint_dir}")
    
    checkpoint_filename = f"checkpoint_step_{global_step}.pth"
    checkpoint_path = os.path.join(run_checkpoint_dir, checkpoint_filename)
    
    state = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_config': model_config,
        'train_config': train_config, # Save current train_config for reference
        'loss': loss 
    }
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path} at epoch {epoch+1}, step {global_step}, loss {loss:.4f}")

    # Manage number of checkpoints
    max_to_keep = train_config.get('max_checkpoints_to_keep', 0)
    if max_to_keep > 0:
        # List all checkpoint files matching the pattern
        checkpoint_files = [
            f for f in os.listdir(run_checkpoint_dir) 
            if f.startswith("checkpoint_step_") and f.endswith(".pth")
        ]
        
        # Parse step numbers and create a list of (step_number, filepath) tuples
        saved_checkpoints_with_steps = []
        for f_name in checkpoint_files:
            try:
                step_num_str = f_name.split('_step_')[1].split('.pth')[0]
                step_num = int(step_num_str)
                saved_checkpoints_with_steps.append((step_num, os.path.join(run_checkpoint_dir, f_name)))
            except (IndexError, ValueError):
                print(f"Warning: Could not parse step number from checkpoint file: {f_name}")
                continue # Skip files that don't match the expected format

        # Sort by step number (ascending, so oldest are first)
        saved_checkpoints_with_steps.sort(key=lambda x: x[0])
        
        if len(saved_checkpoints_with_steps) > max_to_keep:
            num_to_delete = len(saved_checkpoints_with_steps) - max_to_keep
            for i in range(num_to_delete):
                file_to_delete = saved_checkpoints_with_steps[i][1]
                try:
                    os.remove(file_to_delete)
                    print(f"Deleted old checkpoint: {file_to_delete}")
                except OSError as e:
                    print(f"Error deleting old checkpoint {file_to_delete}: {e}")

# Loads model and optimizer state from a specified checkpoint file.
def load_checkpoint(model_class, optimizer_class, checkpoint_path_to_load, device):
    """Loads model and optimizer state from a specific checkpoint_path_to_load."""
    
    if checkpoint_path_to_load is None or not os.path.exists(checkpoint_path_to_load):
        if checkpoint_path_to_load is not None: # Path was given but doesn't exist
            print(f"Specified checkpoint path '{checkpoint_path_to_load}' not found. Will start from scratch.")
        else: # No path was given
            print("No specific checkpoint path specified. Will start from scratch.")
        return None, None, 0, 0 # Model, Optimizer, start_epoch, start_global_step

    print(f"Loading checkpoint from {checkpoint_path_to_load}...")
    checkpoint = torch.load(checkpoint_path_to_load, map_location=device)
    
    loaded_model_config = checkpoint['model_config']
    loaded_train_config = checkpoint.get('train_config', {}) # Get train_config if available
    
    model = model_class(
        vocab_size=loaded_model_config['vocab_size'],
        embed_dim=loaded_model_config['embed_dim'],
        num_q_heads=loaded_model_config['num_q_heads'],
        num_kv_heads=loaded_model_config['num_kv_heads'],
        num_layers=loaded_model_config['num_layers'],
        block_size=loaded_model_config['block_size'],
        dropout_rate=loaded_model_config['dropout_rate'],
        ff_hidden_dim_factor=loaded_model_config.get('ff_hidden_dim_factor', 8/3), # Add default for backward compatibility
        rms_norm_eps=loaded_model_config.get('rms_norm_eps', 1e-5),             # Add default
        rope_base=loaded_model_config.get('rope_base', 10000.0),                  # Add default
        local_attn_window_size=loaded_model_config.get('local_attn_window_size', -1),
        global_every_n_layers=loaded_model_config.get('global_every_n_layers', 0)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Use learning rate from the loaded checkpoint's train_config if available,
    # otherwise, this optimizer won't be used if we are creating a new one later.
    # If 'learning_rate' is not in loaded_train_config, this would error if checkpoint is old.
    # For robustness, one might fall back to a default if creating an optimizer here for a loaded model.
    # However, a proper checkpoint should have this.
    optimizer_lr = loaded_train_config.get('learning_rate', TRAIN_CONFIG['learning_rate']) # Fallback to current TRAIN_CONFIG if not in ckpt
    optimizer = optimizer_class(model.parameters(), lr=optimizer_lr)
    
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint.get('epoch', -1) + 1 
    start_global_step = checkpoint.get('global_step', 0)
    last_loss = checkpoint.get('loss', float('inf'))
    
    print(f"Checkpoint loaded. Resuming from epoch {start_epoch}, global step {start_global_step}. Last loss: {last_loss:.4f}")
    print(f"Loaded model config: {loaded_model_config}")
    if loaded_train_config:
        print(f"Loaded train config (from checkpoint): Learning Rate {optimizer_lr}")

    return model, optimizer, start_epoch, start_global_step


# Try to load a specific checkpoint if path is provided
checkpoint_to_load = TRAIN_CONFIG.get('load_specific_checkpoint_path')

model, optimizer, start_epoch, start_global_step = load_checkpoint(
    SmallGPT, 
    optim.AdamW, 
    checkpoint_to_load, # Pass the specific path (can be None)
    device
)

if model is None: # No checkpoint loaded (or path was None/invalid), so create a new model.
    print("Initializing a new model and optimizer.")
    # Ensure MODEL_CONFIG['vocab_size'] is set (should be by now from tokenizer loading)
    if MODEL_CONFIG['vocab_size'] is None:
        raise ValueError("vocab_size is not set in MODEL_CONFIG. Ensure tokenizer is loaded first.")
        
    model = SmallGPT(
        vocab_size=MODEL_CONFIG['vocab_size'],
        embed_dim=MODEL_CONFIG['embed_dim'],
        num_q_heads=MODEL_CONFIG['num_q_heads'],
        num_kv_heads=MODEL_CONFIG['num_kv_heads'],
        num_layers=MODEL_CONFIG['num_layers'],
        block_size=MODEL_CONFIG['block_size'],
        dropout_rate=MODEL_CONFIG['dropout_rate'],
        ff_hidden_dim_factor=MODEL_CONFIG['ff_hidden_dim_factor'],
        rms_norm_eps=MODEL_CONFIG['rms_norm_eps'],
        rope_base=MODEL_CONFIG['rope_base'],
        local_attn_window_size=MODEL_CONFIG['local_attn_window_size'],
        global_every_n_layers=MODEL_CONFIG['global_every_n_layers']
    )
    optimizer = optim.AdamW(model.parameters(), lr=TRAIN_CONFIG['learning_rate'])
    start_epoch = 0 # Reset epoch and step for new model
    start_global_step = 0

# Compile the model (after instantiation, whether new or loaded)
#print("Compiling the model with torch.compile()...")
#try:
#    model = torch.compile(model)
#    print("Model compiled successfully.")
#except Exception as e:
#    print(f"Failed to compile model: {e}. Proceeding without compilation.")
model.to(device) # Ensure it's on device after potential compilation

# Test forward pass
dummy_batch_x, _ = next(iter(train_loader))
dummy_batch_x = dummy_batch_x.to(device)
with torch.no_grad():
    logits = model(dummy_batch_x)
print(f"Model output logits shape: {logits.shape}") # Expected: (BATCH_SIZE, BLOCK_SIZE, VOCAB_SIZE)

# Count parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")

# Calculate total training steps
total_training_steps = len(train_loader) * TRAIN_CONFIG['num_epochs'] # Use TRAIN_CONFIG
print(f"Total number of training steps: {total_training_steps}")


# %% 5. Training the Model
print("\n--- 5. Training the Model ---\n")

loss_fn = nn.CrossEntropyLoss() # Handles softmax internally
# Optimizer is already defined (either loaded or new)

print(f"Starting/Resuming training from epoch {start_epoch} for {TRAIN_CONFIG['num_epochs']} total epochs...")
model.train() # Set model to training mode

global_step = start_global_step # Initialize global_step from loaded checkpoint
for epoch in range(start_epoch, TRAIN_CONFIG['num_epochs']): # Iterate from start_epoch
    epoch_loss_sum = 0 
    num_batches_in_epoch = 0
    
    batches_in_this_epoch = len(train_loader)

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TRAIN_CONFIG['num_epochs']}", unit="batch")

    current_batch_loss = 0.0 # To store loss for saving checkpoint
    for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(input_ids)
        # Loss calculation expects logits of shape (N*T, V) and targets of shape (N*T)
        loss = loss_fn(logits.view(-1, MODEL_CONFIG['vocab_size']), target_ids.view(-1))
        current_batch_loss = loss.item() # Store current batch loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss_sum += current_batch_loss
        num_batches_in_epoch += 1
        global_step += 1
    
        # Update progress bar on every batch
        progress_bar.set_postfix(loss=current_batch_loss, global_step=global_step)
        
        # --- Intra-epoch checkpoint saving ---
        if TRAIN_CONFIG['save_every_n_steps'] > 0 and global_step % TRAIN_CONFIG['save_every_n_steps'] == 0:
            save_checkpoint(epoch, global_step, model, optimizer, MODEL_CONFIG, TRAIN_CONFIG, current_batch_loss)
            
    avg_epoch_loss = epoch_loss_sum / num_batches_in_epoch if num_batches_in_epoch > 0 else 0
    progress_bar.close() 
    print(f"End of Epoch {epoch+1}/{TRAIN_CONFIG['num_epochs']}, Average Epoch Loss: {avg_epoch_loss:.4f}, Global Steps: {global_step}")

    # Save checkpoint at the end of each epoch using the new mechanism
    # This will also be subject to the max_checkpoints_to_keep rule.
    save_checkpoint(epoch, global_step, model, optimizer, MODEL_CONFIG, TRAIN_CONFIG, avg_epoch_loss)


print("Training complete.")


# %% 6. Simple Text Generation Example
print("\n--- 6. Simple Text Generation Example ---\n")

# Generates text autoregressively using the trained model.
# If you want to load a specific model for generation, you can do it here:
# print("Loading model for generation...")
# loaded_model_for_generation, _, _, _ = load_checkpoint(SmallGPT, optim.AdamW, TRAIN_CONFIG, device)
# if loaded_model_for_generation:
#     # If it was compiled during training, compile again for inference if needed
#     try:
#         print("Compiling loaded model for generation...")
#         loaded_model_for_generation = torch.compile(loaded_model_for_generation)
#         print("Model compiled successfully for generation.")
#     except Exception as e:
#         print(f"Failed to compile loaded model for generation: {e}")
#     loaded_model_for_generation.to(device)
#     loaded_model_for_generation.eval()
#     model_to_generate_with = loaded_model_for_generation
# else:
#     print("Could not load checkpoint for generation, using model from training session.")
#     model.eval() # Ensure the current model is in eval mode
#     model_to_generate_with = model

# For simplicity, we'll just use the model that was in the training session
model.eval() # Ensure model is in evaluation mode
model_to_generate_with = model # Use the model currently in memory (either trained or loaded and then trained)

# The generate_text function needs to be adapted if the model it receives
# is the torch.compile() wrapper. model.eval() should handle it.

def generate_text(model_instance, start_string, tokenizer_instance, max_len=50, temperature=0.8, top_k=None):
    print(f"\nGenerating text from seed: '{start_string}'")
    
    model_instance.eval() # Ensure model is in eval mode
    
    # Encode the starting string using the provided tokenizer
    input_ids = tokenizer_instance.encode(start_string)
    generated_ids = list(input_ids) # Keep track of all generated IDs

    # Ensure input_ids are within block_size for model context
    # The model expects (batch, seq_len)
    current_context_ids = torch.tensor([input_ids[-MODEL_CONFIG['block_size']:]], dtype=torch.long, device=device)

    print("Seed text tokenized IDs:", current_context_ids)
    print("Seed text tokenized IDs shape:", current_context_ids.shape) # For debugging

    with torch.no_grad():
        for i in range(max_len):
            # Get logits from the model
            logits = model_instance(current_context_ids) # (1, current_seq_len, vocab_size)
            
            # Focus on the logits for the last token (next token prediction)
            next_token_logits = logits[:, -1, :] # (1, vocab_size)

            # Apply top-k filtering (optional)
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            
            # Apply temperature scaling to logits
            if temperature > 0: # temperature=0 would make it greedy, but softmax might be an issue.
                                # For pure greedy, use argmax before softmax.
                next_token_logits = next_token_logits / temperature
            
            # Get probabilities using softmax
            probs = F.softmax(next_token_logits, dim=-1) # (1, vocab_size)
            
            # Sample the next token ID
            # Using multinomial sampling; could also use torch.argmax for greedy if temperature is low/zero
            next_token_id = torch.multinomial(probs, num_samples=1).item() # Get scalar
            
            # Append the new token ID
            generated_ids.append(next_token_id)
            
            # Update context: append new token, and trim to block_size if needed
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            current_context_ids = torch.cat((current_context_ids, next_token_tensor), dim=1)
            if current_context_ids.shape[1] > MODEL_CONFIG['block_size']:
                current_context_ids = current_context_ids[:, -MODEL_CONFIG['block_size']:]
            
            # Stop if EOS token is generated
            if tokenizer_instance.eos_token_id is not None and next_token_id == tokenizer_instance.eos_token_id:
                print(f"EOS token ({tokenizer_instance.eos_token_id}) generated at step {i+1}. Stopping.")
                break
            
    return tokenizer_instance.decode(generated_ids) # Use tokenizer to decode the full sequence

# Generate some text (make sure model and tokenizer are defined)
seed_text = "Alice was" # This seed might be out-of-vocabulary more often now
generated_sequence = generate_text(model, seed_text, tokenizer, max_len=100, temperature=0.7, top_k=40)
print("\nGenerated Text 1:")
print(generated_sequence)

seed_text_2 = "The quick brown fox"
generated_sequence_2 = generate_text(model, seed_text_2, tokenizer, max_len=100, temperature=0.7, top_k=40)
print("\nGenerated Text 2:")
print(generated_sequence_2)


print("\nEnd of Module 18.")
