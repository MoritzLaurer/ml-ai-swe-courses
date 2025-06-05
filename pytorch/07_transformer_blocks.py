# -*- coding: utf-8 -*-
# 07_transformer_blocks.py

# Module 7: Building Transformer Blocks for LLMs
#
# This script covers implementation of the core components of a Transformer block:
# 1. Scaled Dot-Product Attention (the core attention mechanism).
# 2. Multi-Head Attention (`nn.Module` wrapping scaled dot-product).
# 3. Position-wise Feed-Forward Network (`nn.Module`).
# 4. Residual Connections and Layer Normalization (Pre-LN variant shown).
# 5. Creating a Causal (Look-ahead) Mask for decoder self-attention.
# 6. Combining components into a Transformer Decoder Block (`nn.Module`).
# 7. Brief overview of Positional Encoding.
# 8. Comparisons to JAX implementations.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("--- Module 7: Building Transformer Blocks for LLMs ---\n")
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


# %% 1. Scaled Dot-Product Attention
print("\n--- 1. Scaled Dot-Product Attention ---\n")

# Formula: Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V
# Where Q=Query, K=Key, V=Value, d_k = dimension of Key vectors.
# The mask is typically applied before the softmax.

def scaled_dot_product_attention(query, key, value, mask=None, dropout_p=0.0):
    """
    Computes scaled dot-product attention.

    Args:
        query (torch.Tensor): Query tensor; shape (batch, ..., seq_len_q, depth)
        key (torch.Tensor): Key tensor; shape (batch, ..., seq_len_k, depth)
        value (torch.Tensor): Value tensor; shape (batch, ..., seq_len_v, depth)
                                Note: seq_len_k == seq_len_v
        mask (torch.Tensor, optional): Boolean mask; shape (..., seq_len_q, seq_len_k).
                                       True indicates position should be masked (ignored). Defaults to None.
        dropout_p (float, optional): Dropout probability. Defaults to 0.0.

    Returns:
        torch.Tensor: Output tensor after attention; shape (batch, ..., seq_len_q, depth)
        torch.Tensor: Attention weights; shape (batch, ..., seq_len_q, seq_len_k)
    """
    embed_dim = query.size(-1)
    # Calculate raw attention scores (QK^T)
    # (batch, ..., seq_len_q, depth) @ (batch, ..., depth, seq_len_k) -> (batch, ..., seq_len_q, seq_len_k)
    scores = torch.matmul(query, key.transpose(-2, -1))

    # Scale the scores by sqrt(d_k)
    scale_factor = math.sqrt(embed_dim)
    scaled_scores = scores / scale_factor

    # Apply mask (if provided) before softmax
    if mask is not None:
        # Masked positions are filled with a large negative value (-inf)
        # so that softmax assigns them near-zero probability.
        # `masked_fill_` operates in-place. Use `masked_fill` for functional style.
        # The mask shape needs to broadcast correctly. Typical shapes:
        # Padding Mask: (batch, 1, 1, seq_len_k)
        # Causal Mask: (1, 1, seq_len_q, seq_len_k)
        scaled_scores = scaled_scores.masked_fill(mask == True, float('-inf')) # True means MASKED position

    # Apply softmax to get attention weights
    attn_weights = F.softmax(scaled_scores, dim=-1)

    # Apply dropout to attention weights (optional)
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # Multiply weights by values (Attention * V)
    # (batch, ..., seq_len_q, seq_len_k) @ (batch, ..., seq_len_v, depth) -> (batch, ..., seq_len_q, depth)
    # Note: seq_len_k == seq_len_v
    output = torch.matmul(attn_weights, value)

    return output, attn_weights

# --- Example Usage ---
batch_size, seq_len, embed_dim = 2, 5, 8
q_dummy = torch.randn(batch_size, seq_len, embed_dim, device=device)
k_dummy = torch.randn(batch_size, seq_len, embed_dim, device=device)
v_dummy = torch.randn(batch_size, seq_len, embed_dim, device=device)

print("Testing Scaled Dot-Product Attention:")
attn_output, attn_weights = scaled_dot_product_attention(q_dummy, k_dummy, v_dummy)
print(f"  Input Q/K/V shape: {q_dummy.shape}")
print(f"  Attention Output shape: {attn_output.shape}")
print(f"  Attention Weights shape: {attn_weights.shape}")
print(f"  Input Q: {q_dummy}")
print(f"  Attention Output: {attn_output}")
print(f"  Attention Weights: {attn_weights}")


# %% 2. Multi-Head Attention (`nn.Module`)
print("\n--- 2. Multi-Head Attention (`nn.Module`) ---\n")

# Runs scaled dot-product attention multiple times in parallel ("heads").
# 1. Linearly project Q, K, V for each head.
# 2. Apply scaled dot-product attention for each head.
# 3. Concatenate results from all heads.
# 4. Apply a final linear projection.

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        # Linear layers for Q, K, V projections (can be combined into one large layer for efficiency)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Final output linear layer
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Optional: Implement custom weight initialization if desired
        # Default nn.Linear init is often reasonable (Kaiming uniform)
        pass

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query (torch.Tensor): Query tensor; shape (batch, seq_len_q, embed_dim)
            key (torch.Tensor): Key tensor; shape (batch, seq_len_k, embed_dim)
            value (torch.Tensor): Value tensor; shape (batch, seq_len_v, embed_dim)
            mask (torch.Tensor, optional): Boolean attention mask. Defaults to None.

        Returns:
            torch.Tensor: Output tensor; shape (batch, seq_len_q, embed_dim)
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        seq_len_v = value.size(1)

        # 1. Linear projections
        q = self.q_proj(query) # (batch, seq_len_q, embed_dim)
        k = self.k_proj(key)   # (batch, seq_len_k, embed_dim)
        v = self.v_proj(value) # (batch, seq_len_v, embed_dim)

        # 2. Reshape and transpose for multi-head attention
        # Reshape from (batch, seq_len, embed_dim) to (batch, seq_len, num_heads, head_dim)
        # Transpose to (batch, num_heads, seq_len, head_dim) for attention calculation
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_v, self.num_heads, self.head_dim).transpose(1, 2)

        # Adjust mask shape if needed for broadcasting over heads
        # Expected mask shape for attention function: (batch, num_heads, seq_len_q, seq_len_k) or broadcastable
        if mask is not None:
            # Add head dimension if mask is (batch, seq_len_q, seq_len_k) or (seq_len_q, seq_len_k)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1) # Add head dim -> (batch, 1, seq_len_q, seq_len_k)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0) # Add batch and head dim -> (1, 1, seq_len_q, seq_len_k)
            # else assume mask is already correctly shaped (e.g., from create_causal_mask)

        # 3. Scaled dot-product attention for all heads in parallel
        # attn_output shape: (batch, num_heads, seq_len_q, head_dim)
        # attn_weights shape: (batch, num_heads, seq_len_q, seq_len_k) - we ignore weights here
        attn_output, _ = scaled_dot_product_attention(q, k, v, mask=mask, dropout_p=self.dropout)

        # 4. Concatenate heads and reshape back
        # Transpose back: (batch, seq_len_q, num_heads, head_dim)
        # Use .contiguous() before .view() if memory layout changed due to transpose
        attn_output = attn_output.transpose(1, 2).contiguous()
        # Reshape: (batch, seq_len_q, embed_dim)
        attn_output = attn_output.view(batch_size, seq_len_q, self.embed_dim)

        # 5. Final linear projection
        output = self.out_proj(attn_output)

        return output


# --- Example Usage ---
mha_embed_dim = 64
mha_num_heads = 4
mha_seq_len = 10
mha_batch_size = 2

mha = MultiHeadAttention(mha_embed_dim, mha_num_heads, dropout=0.1).to(device)
q_mha = torch.randn(mha_batch_size, mha_seq_len, mha_embed_dim, device=device)
k_mha = torch.randn(mha_batch_size, mha_seq_len, mha_embed_dim, device=device)
v_mha = torch.randn(mha_batch_size, mha_seq_len, mha_embed_dim, device=device)

print("\nTesting Multi-Head Attention:")
mha_output = mha(q_mha, k_mha, v_mha)
print(f"  Input Q/K/V shape: {q_mha.shape}")
print(f"  Output shape: {mha_output.shape}")


# %% 3. Position-wise Feed-Forward Network (`nn.Module`)
print("\n--- 3. Position-wise Feed-Forward Network (`nn.Module`) ---\n")

# Two linear layers with an activation in between. Applied independently to each position.
# FFN(x) = Linear_2( Activation( Linear_1(x) ) )
# Often expands dimensionality in the intermediate layer (e.g., 4x embed_dim).

class PositionwiseFeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout=0.1, activation=F.relu):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        # output shape: (batch, seq_len, embed_dim)
        return x

# --- Example Usage ---
ffn_embed_dim = 64
ffn_hidden_dim = ffn_embed_dim * 4 # Common expansion factor
ffn_seq_len = 10
ffn_batch_size = 2

ffn = PositionwiseFeedForward(ffn_embed_dim, ffn_hidden_dim).to(device)
ffn_input = torch.randn(ffn_batch_size, ffn_seq_len, ffn_embed_dim, device=device)

print("Testing Position-wise Feed-Forward Network:")
ffn_output = ffn(ffn_input)
print(f"  Input shape: {ffn_input.shape}")
print(f"  Output shape: {ffn_output.shape}")


# %% 4. Residual Connections and Layer Normalization
print("\n--- 4. Residual Connections and Layer Normalization ---\n")

# Residual Connection: output = x + Sublayer(x)
# Layer Normalization (`nn.LayerNorm`): Normalizes across the embedding dimension.

# Two main variants:
# 1. Post-LN: LayerNorm(x + Dropout(Sublayer(x))) - Original Transformer paper
# 2. Pre-LN: x + Dropout(Sublayer(LayerNorm(x))) - Often more stable training

# We will implement the Pre-LN version in the Decoder Block below.
# LayerNorm instance: `nn.LayerNorm(embed_dim)`

print("Pre-LN Structure (conceptual):")
print("  norm_out = LayerNorm(x)")
print("  sublayer_out = Sublayer(norm_out)")
print("  output = x + Dropout(sublayer_out)")


# %% 5. Creating a Causal (Look-ahead) Mask
print("\n--- 5. Creating a Causal (Look-ahead) Mask ---\n")

# For decoder self-attention, a position should only attend to previous positions
# and itself. We need a mask to prevent attention to future positions.

def create_causal_mask(seq_len, device):
    """
    Creates a causal mask for decoder self-attention.

    Args:
        seq_len (int): The length of the sequence.
        device: The torch device.

    Returns:
        torch.Tensor: Boolean mask of shape (1, 1, seq_len, seq_len).
                      True indicates a masked (forbidden) position.
    """
    # Create a square matrix with True below the main diagonal (and on the diagonal).
    # Use triu (upper triangle) with diagonal=1 to get True for future positions.
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    # Reshape to (1, 1, seq_len, seq_len) for broadcasting with attention scores (batch, heads, q, k)
    return mask.unsqueeze(0).unsqueeze(0)

# --- Example Usage ---
mask_seq_len = 5
causal_mask = create_causal_mask(mask_seq_len, device)
print(f"Causal mask for seq_len={mask_seq_len} (shape {causal_mask.shape}):")
# Print the mask content (squeeze to 2D for readability)
# True means MASKED (cannot attend)
print(causal_mask.squeeze())


# %% 6. Combining into a Transformer Decoder Block (`nn.Module`)
print("\n--- 6. Combining into a Transformer Decoder Block (`nn.Module`) ---\n")

# Structure (GPT-style Decoder, Pre-LN):
# 1. LayerNorm -> Masked Multi-Head Self-Attention -> Dropout -> Add (Residual)
# 2. LayerNorm -> Position-wise Feed-Forward -> Dropout -> Add (Residual)

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = PositionwiseFeedForward(embed_dim, ffn_dim, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None):
        """
        Args:
            x (torch.Tensor): Input tensor; shape (batch, seq_len, embed_dim)
            causal_mask (torch.Tensor, optional): Causal mask for self-attention;
                                           shape (1, 1, seq_len, seq_len) or broadcastable.
                                           Ensures attention is only to previous positions. Defaults to None.

        Returns:
            torch.Tensor: Output tensor; shape (batch, seq_len, embed_dim)
        """
        # 1. Masked Multi-Head Self-Attention (Pre-LN)
        residual = x
        x_norm = self.norm1(x)
        # In a decoder, the self_attn is typically masked to be causal
        attn_output = self.self_attn(query=x_norm, key=x_norm, value=x_norm, mask=causal_mask)
        x = residual + self.dropout1(attn_output)

        # 2. Position-wise Feed-Forward (Pre-LN)
        residual = x
        x_norm = self.norm2(x)
        ffn_output = self.feed_forward(x_norm)
        x = residual + self.dropout2(ffn_output)

        return x

# --- Example Usage ---
block_embed_dim = 64
block_num_heads = 8
block_ffn_dim = block_embed_dim * 4
block_seq_len = 10
block_batch_size = 2

decoder_block = DecoderBlock(block_embed_dim, block_num_heads, block_ffn_dim).to(device)
block_input = torch.randn(block_batch_size, block_seq_len, block_embed_dim, device=device)
# For a decoder block, a causal mask is essential for auto-regressive behavior
block_causal_mask = create_causal_mask(block_seq_len, device)

print("Testing Decoder Block:")
block_output = decoder_block(block_input, causal_mask=block_causal_mask)
print(f"  Input shape: {block_input.shape}")
print(f"  Causal mask shape: {block_causal_mask.shape}")
print(f"  Output shape: {block_output.shape}")


# %% 6.5. Transformer Encoder Block (`nn.Module`) - Example for BERT-like models
print("\n--- 6.5. Transformer Encoder Block (`nn.Module`) ---\n")

# Structure (BERT-style Encoder, Pre-LN):
# 1. LayerNorm -> Bi-directional Multi-Head Self-Attention -> Dropout -> Add (Residual)
# 2. LayerNorm -> Position-wise Feed-Forward -> Dropout -> Add (Residual)
# The key difference from the DecoderBlock is that the self-attention is typically NOT causally masked.
# It might receive a padding_mask to ignore padding tokens in batched sequences.
# The only small differences to the DecoderBlock are highlighted with comments below.

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        # The same MultiHeadAttention module can be used, the difference is in the mask passed.
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = PositionwiseFeedForward(embed_dim, ffn_dim, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None): # DecoderBlock uses 'causal_mask' here. Mask here is typically for padding
        """
        Args:
            x (torch.Tensor): Input tensor; shape (batch, seq_len, embed_dim)
            padding_mask (torch.Tensor, optional): Padding mask for self-attention;
                                           shape (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
                                           or broadcastable. True indicates a PAD token to be ignored.
                                           Defaults to None (no masking).
                                           DIFFERENCE: DecoderBlock expects a 'causal_mask' to prevent attending to future tokens.

        Returns:
            torch.Tensor: Output tensor; shape (batch, seq_len, embed_dim)
        """
        # 1. Multi-Head Self-Attention (Pre-LN)
        residual = x
        x_norm = self.norm1(x)
        # DIFFERENCE: In an encoder, self_attn allows attending to all tokens (bidirectional).
        # A padding_mask might be used if input sequences have padding.
        # DecoderBlock passes 'causal_mask' here to enforce auto-regressive behavior.
        attn_output = self.self_attn(query=x_norm, key=x_norm, value=x_norm, mask=padding_mask)
        x = residual + self.dropout1(attn_output)

        # 2. Position-wise Feed-Forward (Pre-LN)
        residual = x
        x_norm = self.norm2(x)
        ffn_output = self.feed_forward(x_norm)
        x = residual + self.dropout2(ffn_output)

        return x

# --- Example Usage for EncoderBlock ---
enc_block_embed_dim = 64
enc_block_num_heads = 8
enc_block_ffn_dim = enc_block_embed_dim * 4
enc_block_seq_len = 12
enc_block_batch_size = 3

encoder_block = EncoderBlock(enc_block_embed_dim, enc_block_num_heads, enc_block_ffn_dim).to(device)
enc_block_input = torch.randn(enc_block_batch_size, enc_block_seq_len, enc_block_embed_dim, device=device)

# Example of a padding mask (e.g., last 2 tokens in each sequence are padding)
# Mask shape (batch, 1, 1, seq_len_k) for scaled_dot_product_attention
# True means MASKED position
enc_padding_mask = torch.zeros(enc_block_batch_size, 1, 1, enc_block_seq_len, device=device, dtype=torch.bool)
if enc_block_seq_len > 2:
    enc_padding_mask[:, :, :, -2:] = True # Mask the last two tokens

print("Testing Encoder Block:")
# No causal mask is typically passed to an encoder block's self-attention.
# A padding mask might be passed if sequences are padded.
enc_block_output_no_mask = encoder_block(enc_block_input, padding_mask=None)
enc_block_output_with_padding_mask = encoder_block(enc_block_input, padding_mask=enc_padding_mask)

print(f"  Input shape: {enc_block_input.shape}")
print(f"  Output shape (no mask): {enc_block_output_no_mask.shape}")
print(f"  Padding mask example shape: {enc_padding_mask.shape if enc_padding_mask is not None else 'None'}")
print(f"  Padding mask example content (first batch): \n{enc_padding_mask[0].squeeze() if enc_padding_mask is not None else 'None'}")
print(f"  Output shape (with padding mask): {enc_block_output_with_padding_mask.shape}")


# %% 7. Positional Encoding (Brief Overview)
print("\n--- 7. Positional Encoding (Brief Overview) ---\n")

# Transformers themselves don't know the order of tokens. We need to inject position info.
# Two common methods:
# 1. Learned Positional Embeddings: An `nn.Embedding` layer where the input is position index (0, 1, ..., seq_len-1). Added to token embeddings.
# 2. Fixed Sinusoidal Positional Encoding: Uses sin/cos functions of different frequencies based on position and embedding dimension. No learnable parameters. Added to token embeddings.

# --- Sinusoidal Formula Sketch (for Additive Sinusoidal Encoding) ---
# PE(pos, 2i)   = sin(pos / 10000^(2i / embed_dim))
# PE(pos, 2i+1) = cos(pos / 10000^(2i / embed_dim))
# Where 'pos' is the position index, 'i' is the dimension index within the embedding.

# Implementation often involves pre-calculating these values into a tensor
# of shape (max_seq_len, embed_dim) and adding it to the input embeddings.

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)) # (embed_dim / 2)
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Add batch dimension -> (1, max_len, embed_dim)
        # Register as buffer so it's part of the state_dict but not a parameter
        self.register_buffer('pe', pe, persistent=False) # False if you don't need to save/load it explicitly

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Add positional encoding up to the sequence length of x
        # Assumes x embeddings are already created.
        x = x + self.pe[:, :x.size(1)]
        return x

# Usage: Typically applied once after the token embedding layer.
# pos_encoder = SinusoidalPositionalEncoding(embed_dim, max_len=512).to(device)
# token_embeddings = embedding_layer(input_ids)
# input_with_pos = pos_encoder(token_embeddings)
# output = decoder_block(input_with_pos, mask=causal_mask)


# 3. Rotary Positional Embeddings (RoPE) - A more recent approach
print("\n--- 7.1 Rotary Positional Embeddings (RoPE) ---\n")
print("RoPE is another method for injecting positional information, popular in models like LLaMA.")
print("Key differences from additive sinusoidal or learned embeddings:")
print("- Relative Positions: RoPE is designed to encode relative positional information directly into the attention mechanism.")
print("  The attention score between a query at position 'm' and a key at position 'n' becomes sensitive")
print("  to their relative distance (m-n), rather than just their absolute positions.")
print("- Application: Instead of adding positional vectors to token embeddings, RoPE modifies (rotates)")
print("  the query (Q) and key (K) vectors within the attention calculation, just before the QK^T dot product.")
print("  This rotation is based on the token's absolute position but results in relative positional awareness in the dot product.")
print("- No Additive Component: Typically, no separate positional embedding is added to the input token embeddings.")
print("- Properties: Often cited for better length extrapolation and naturally decaying attention with distance.")
print("- Implementation: Involves applying 2D rotation matrices (derived from sines and cosines based on position")
print("  and feature dimension) to pairs of features within the Q and K vectors.")


# %% 8. JAX Comparison
print("\n--- 8. JAX Comparison ---\n")

print("Implementing Transformer blocks in JAX/Flax/Haiku:")
print("- Functional Style: Core logic like scaled dot-product attention is often a standalone function.")
print("- Explicit State: Modules (like Flax `linen.Module`) manage parameters, but these are passed explicitly during the `apply` call.")
print("- Masking: Mask arguments are passed explicitly to attention functions/modules.")
print("- PRNG Handling: Dropout requires explicit PRNG key handling in JAX for reproducibility.")
print("- Performance: JIT compilation (`@jax.jit`) can optimize the entire block's computation graph, potentially leading to performance gains, especially on TPUs.")
print("- Libraries: Flax and Haiku provide pre-built layers (Dense, LayerNorm, MultiHeadDotProductAttention) simplifying implementation.")


# %% Conclusion
print("\n--- Module 7 Summary ---\n")
print("Key Takeaways:")
print("- Core Transformer components: Scaled Dot-Product Attention, Multi-Head Attention, Position-wise FFN.")
print("- These are combined with Residual Connections and Layer Normalization (Pre-LN is common) to form blocks.")
print("- Decoder blocks use Causal (Look-ahead) Masks in self-attention to prevent seeing future tokens.")
print("- Positional Encoding is necessary to inform the model about token order.")
print("- PyTorch provides `nn.Module` building blocks for implementing these components.")

print("\nEnd of Module 7.")