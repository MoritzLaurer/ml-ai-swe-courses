import math
import torch
import torch.nn as nn

# RMSNorm implementation (as used in LLaMA, PaLM, etc.)
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # Scale parameter vector
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        # x: (batch, seq, dim)
        # Compute RMS over last dimension
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed

class RotaryEmbedding:
    """Pre-compute cosine and sine frequencies for RoPE."""
    def __init__(self, dim, max_seq_len=2048):
        # dim is half of the rotary embedding dimension (since we use pairs)
        self.dim = dim
        # Define the base frequencies (as in RoPE paper) 
        # Use exponent spacing: base^(2i/dim)
        theta = 10000 ** (-torch.arange(0, dim, 2).float() / dim)
        # theta: (dim/2,) frequencies for alternating pairs
        seq = torch.arange(max_seq_len).float()
        # Outer product to get rotation angles for each position
        angles = torch.outer(seq, theta)  # shape (max_seq_len, dim/2)
        # Compute cosines and sines
        self.cos = torch.cos(angles)  # (max_seq_len, dim/2)
        self.sin = torch.sin(angles)  # (max_seq_len, dim/2)
        # Register buffers (so they move to GPU with the model if needed)
        self.cos = self.cos.cuda() if torch.cuda.is_available() else self.cos
        self.sin = self.sin.cuda() if torch.cuda.is_available() else self.sin

    def apply_rotary(self, q: torch.Tensor, k: torch.Tensor, start_index: int = 0):
        """Apply rotary position embedding to the first self.dim components of q and k."""
        # q, k: (batch, seq_len, n_heads, head_dim)
        seq_len = q.shape[1]
        cos = self.cos[start_index: start_index + seq_len]  # (seq_len, dim/2)
        sin = self.sin[start_index: start_index + seq_len]  # (seq_len, dim/2)
        # Align shapes for broadcasting: (batch, seq_len, n_heads, dim/2)
        cos = cos[None, :, None, :]  # add batch and head dims
        sin = sin[None, :, None, :]
        # Split the first part of q,k into two halves (for real and imaginary parts)
        q_rot = q[..., :self.dim]    # (..., dim)
        q_pass = q[..., self.dim:]  # remainder (if head_dim > rotary dim)
        k_rot = k[..., :self.dim]
        k_pass = k[..., self.dim:]
        # Rotate (x,y) by angle: (x*cos - y*sin, x*sin + y*cos)
        # Here, split q_rot into [x, y] pairs along last axis
        # Note: dim is even by construction
        x_q, y_q = q_rot[..., 0::2], q_rot[..., 1::2]
        x_k, y_k = k_rot[..., 0::2], k_rot[..., 1::2]
        # Apply rotation
        q_rotated_x = x_q * cos - y_q * sin
        q_rotated_y = x_q * sin + y_q * cos
        k_rotated_x = x_k * cos - y_k * sin
        k_rotated_y = x_k * sin + y_k * cos
        # Interleave the rotated components back together
        # (We reversed the splitting of even/odd indices)
        q_rotated = torch.stack([q_rotated_x, q_rotated_y], dim=-1).flatten(-2)
        k_rotated = torch.stack([k_rotated_x, k_rotated_y], dim=-1).flatten(-2)
        # Concatenate the unrotated part back (if any)
        q_out = torch.cat([q_rotated, q_pass], dim=-1)
        k_out = torch.cat([k_rotated, k_pass], dim=-1)
        return q_out, k_out

class MultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_kv_heads=None, rotary_pct=1.0, max_seq_len=2048):
        """
        Multi-head self-attention with optional Grouped-Query (if num_kv_heads < num_heads).
        rotary_pct: fraction of head_dim to apply rotary embedding to (typically 1.0 or 0.5).
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads  # default: no grouping
        assert self.num_heads % self.num_kv_heads == 0, "num_heads must be multiple of num_kv_heads"
        self.head_dim = embed_dim // num_heads
        # Define linear projections
        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        # Rotary embedding utility
        # We'll apply rotary to either full head_dim or half (common is 50%, i.e. rotary on even-indexed dims).
        rotary_dim = int(self.head_dim * rotary_pct)
        # Ensure even dimension for pairing:
        if rotary_dim % 2 != 0:
            rotary_dim -= 1
        self.rotary = RotaryEmbedding(rotary_dim // 2, max_seq_len) if rotary_dim > 0 else None
        self.rotary_dim = rotary_dim

    def forward(self, x, start_pos=0):
        """
        x: Tensor of shape (batch, seq_len, embed_dim)
        start_pos: starting position index (for rotary embedding offset, useful in decoding).
        """
        B, T, _ = x.shape
        # Project to multi-head Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        # Apply rotary positional embedding to q and k (for first rotary_dim components)
        if self.rotary is not None and self.rotary_dim > 0:
            q[..., :self.rotary_dim], k[..., :self.rotary_dim] = self.rotary.apply_rotary(
                q[..., :self.rotary_dim], k[..., :self.rotary_dim], start_index=start_pos
            )
        # If GQA is in effect (grouped queries):
        if self.num_kv_heads != self.num_heads:
            # Repeat k, v across query head groups
            # Each KV head is shared by group_size query heads
            group_size = self.num_heads // self.num_kv_heads
            # Repeat along head dimension
            k = k.unsqueeze(3).expand(-1, -1, -1, group_size, -1)  # shape: (B, T, n_kv, group_size, head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, group_size, -1)
            # Merge the group dimension into the head dimension
            k = k.reshape(B, T, self.num_heads, self.head_dim)
            v = v.reshape(B, T, self.num_heads, self.head_dim)
        else:
            # If no grouping, just repeat K,V for each head (happens implicitly if num_kv_heads == num_heads)
            k = k.view(B, T, self.num_heads, self.head_dim)
            v = v.view(B, T, self.num_heads, self.head_dim)
        # Now q, k, v all have shape (B, T, num_heads, head_dim)
        # Compute scaled dot-product attention
        # Rearrange to (B, num_heads, T, head_dim) for easier matmul
        q = q.permute(0, 2, 1, 3)  # (B, num_heads, seq, head_dim)
        k = k.permute(0, 2, 1, 3)  # (B, num_heads, seq, head_dim)
        v = v.permute(0, 2, 1, 3)  # (B, num_heads, seq, head_dim)
        # Scaled dot products
        att_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, num_heads, seq, seq)
        # Causal mask: prevent attention to future tokens
        # We create a lower-triangular mask of shape (1,1,seq,seq)
        mask = torch.full((T, T), float('-inf'), device=x.device)
        mask = torch.triu(mask, diagonal=1)  # upper triangular (future positions) set to -inf
        att_scores = att_scores + mask  # broadcast over batch and heads
        # Softmax over the last dimension (keys)
        att_prob = torch.softmax(att_scores, dim=-1)  # (B, num_heads, seq, seq)
        # Weighted sum of values
        out = torch.matmul(att_prob, v)  # (B, num_heads, seq, head_dim)
        # Rearrange back to (B, seq, num_heads, head_dim)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(B, T, self.num_heads * self.head_dim)  # concat heads
        # Final linear projection
        out = self.o_proj(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_kv_heads=None, ffn_dim=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn = MultiheadSelfAttention(embed_dim, num_heads, num_kv_heads=num_kv_heads)
        # Use RMSNorm for normalization (pre-normalization as in many LLMs)
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)
        # Feed-forward network
        inner_dim = ffn_dim if ffn_dim is not None else 4 * embed_dim  # expand factor (4x by default)
        # Using SwiGLU activation: implement as Linear -> split -> SiLU & multiply -> Linear
        self.ffn_pre = nn.Linear(embed_dim, 2 * inner_dim)  # two halves for gating
        self.ffn_post = nn.Linear(inner_dim, embed_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        # x: (batch, seq, embed_dim)
        # Self-attention with skip connection
        attn_out = self.attn(self.norm1(x))
        x = x + attn_out  # residual connection
        # Feed-forward with skip connection
        ffn_in = self.norm2(x)
        # Gated FFN: first linear produces 2*inner_dim vector, split into [gate, transform]
        ffn_hidden = self.ffn_pre(ffn_in)  # shape (batch, seq, 2*inner_dim)
        inner = ffn_hidden[..., :ffn_hidden.shape[-1]//2]
        gate = ffn_hidden[..., ffn_hidden.shape[-1]//2:]
        ffn_out = self.ffn_post(self.act(inner) * gate)
        x = x + ffn_out
        return x

class GPTModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_kv_heads=None, num_layers=4, max_seq_len=2048):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = None  # not used since we use rotary
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, num_kv_heads=num_kv_heads)
            for _ in range(num_layers)
        ])
        self.norm_final = RMSNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size, bias=False)
    
    def forward(self, input_ids):
        """
        input_ids: Tensor[int] shape (batch, seq_len)
        Returns logits of shape (batch, seq_len, vocab_size).
        """
        B, T = input_ids.shape
        # Token embeddings
        x = self.token_emb(input_ids)  # (B, T, embed_dim)
        # We rely on rotary embeddings inside attention, so no explicit positional embedding addition here.
        for layer in self.layers:
            x = layer(x)  # each block applies its own RMSNorm, attention, and FFN
        x = self.norm_final(x)
        logits = self.output_proj(x)  # (B, T, vocab_size)
        return logits
