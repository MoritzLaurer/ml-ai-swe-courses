### 3. Local–Global Attention Layers (Sliding Window Strategy) in Modern Models

Many recent large models (such as Google's Gemma 3 and Cohere's Command-A) adopt an interleaved attention pattern: they alternate between local (sliding-window) attention layers and global (full) attention layers. This design is a response to the challenges of scaling context length and memory usage in Transformers. By using mostly local attention and only occasional global attention, the model can handle long sequences much more efficiently while preserving the ability to capture long-range dependencies.

**Illustration of attention layer patterns in Gemma models**
*Source: [developers.googleblog.com](https://developers.googleblog.com)*

Gemma 1 used all-global layers (every layer attends to the full sequence). Gemma 2 introduced alternating global and local layers (blue = local). Gemma 3 goes further, using blocks of 5 local-attention layers per 1 global-attention layer.

**Rationale and Benefits:**
In a full self-attention layer, each token attends to all \(N\) tokens before it (O(\(N^2\)) complexity). This becomes prohibitively expensive for long sequences (both in compute and memory). Local (sliding window) attention restricts each token's attention to a window of the most recent \(w\) tokens (for causal LMs) or neighboring tokens (for bidirectional contexts). This brings complexity down to O(\(N \cdot w\)), which is linear in sequence length for fixed window size. For example, if \(w=1024\) and \(N=16384\), local attention is 16× cheaper than global in that layer. Recent research found that you can replace many full attention layers with local ones and still maintain model quality ([developers.googleblog.com](https://developers.googleblog.com), [ritvik19.medium.com](https://ritvik19.medium.com)). Short-range patterns (syntax, local coherency, recent tokens) are handled by the local attentions, and then an occasional global layer allows information to mix across the whole sequence to handle long-range dependencies or global context.

The trade-off here is a balance between performance and modeling power. By using fewer global-attention layers, the model saves a huge amount of memory and compute – especially for very long contexts. Not every layer needs to pay the cost of attending across 100k tokens; maybe 1 in 6 layers does, and the rest focus on a smaller neighborhood. Google reports that in Gemma 3 (with a 128k token context), this 5:1 local:global ratio cut KV cache memory requirements down to about one-sixth of what they would be with all-global layers ([developers.googleblog.com](https://developers.googleblog.com), [developers.googleblog.com](https://developers.googleblog.com)). In other words, interleaving attention layers "decreases memory requirements, enabling support for an extended context length" ([developers.googleblog.com](https://developers.googleblog.com)). Command-A, with a 256k context, similarly uses a 3:1 ratio of local to full attention to make that feasible ([cohere.com](https://cohere.com)). Without these techniques, storing keys/values for 256k tokens in every layer would be infeasible on today's hardware. Using mostly sliding windows means that for most layers you only need to store the past \(w\) tokens' keys, not all \(N\) tokens – a dramatic reduction in cache size and attention computation.

**Model quality impact:**
Impressively, this strategy has minimal impact on accuracy when done properly. For instance, Gemma 3's 5:1 scheme achieved virtually the same perplexity as if all layers were global (Gemma 2's 1:1 scheme) ([developers.googleblog.com](https://developers.googleblog.com), [ritvik19.medium.com](https://ritvik19.medium.com)). The intuition is that as long as you regularly insert a full-attention layer, the model can propagate information from distant tokens through those global layers. A token might not directly attend to a far token in a local layer, but after a global layer, it can indirectly get that information. With multiple Transformer blocks, even alternating local/global can transmit context across the network. Empirical results show that e.g. one global layer for every 3–6 local layers retains strong performance on language tasks, while massively cutting resource usage. The trade-off is that very long-distance interactions might take an extra layer or two to be integrated (adding a tiny bit of "lag" in how information travels up the layers), and the model architecture is more complex (mix of layer types). But for the gain in efficiency, it's well worth it. Another minor downside is implementation complexity: you need to maintain two kinds of attention layers and possibly tune hyperparameters like window size or the ratio of layers. Also, handling positional embeddings can get tricky – e.g. Command-A uses rotary embeddings (RoPE) in local layers but no positional encoding in global layers ([cohere.com](https://cohere.com)), and Gemma 3 uses different RoPE scaling for local vs global layers ([ritvik19.medium.com](https://ritvik19.medium.com)). These details are to ensure the model can still handle extremely long sequences without position encoding issues. For an educational model, you might not need such complications unless you truly push context length to very high values.

**Implementing alternating local/global layers:**
In your model, this would mean modifying the Transformer block sequence. For example, if you have `num_layers` Transformer layers, you could decide that every 4th layer (or 6th layer, etc.) is a "global" attention layer, and the others are "local" attention layers. Local attention layer means the self-attention is constrained to a sliding window of size \(w\) (plus it's causal, so a token only attends to the previous \(w\) tokens). Global layer is a normal full self-attention (all previous tokens, as usual in a Transformer). You can implement a local sliding window attention in PyTorch by using an attention mask that masks out tokens outside the window.

Here's how you can construct a sliding window attention mask for causal local attention of window size W (where each token can attend to itself and the previous W-1 tokens only):

```python
import torch.nn.functional as F

def sliding_window_attention(q, k, v, window_size: int):
    """
    Perform scaled dot-product attention with a causal sliding window mask.
    q, k, v: shape [batch, n_heads, seq_len, head_dim]
    """
    L = q.size(-2)  # sequence length
    # Create an (L x L) boolean mask where True = allowed to attend
    idx = torch.arange(L, device=q.device)
    # allowed[i,j] = True if j is within window (i-W < j <= i)
    allowed = (idx[:, None] - idx[None, :] < window_size) & (idx[:, None] - idx[None, :] >= 0)
    # Use the mask in PyTorch's scaled_dot_product_attention
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, is_causal=False)
    return out  # shape [batch, n_heads, L, head_dim]
```

In this function, `allowed[i,j]` will be True only if token `j` is within the last `window_size` tokens up to token `i`. We include `j == i` (the token itself) and up to `W-1` tokens behind it. We also ensure causal direction by disallowing any case where `j > i` (the `difference < 0` check ensures we don't attend to future tokens). We pass `is_causal=False` because our custom mask already enforces causality. This uses PyTorch's SDPA for the actual computation – which on MPS will do the masked matmul under the hood. You could also achieve the same by constructing a mask tensor and feeding it to `nn.MultiheadAttention` (noting that `MultiheadAttention` expects a mask where `True` means forbidden position in its current implementation, the inverse of the above). The key point is that the mask is banded: each query position can only see a fixed number of previous positions.

To integrate this into your model, you might set up your Transformer layers like:

```python
# Pseudocode for alternating local vs global layers
for layer_idx, layer in enumerate(transformer_layers):
    if layer_idx % 6 == 5:
        # Every 6th layer (e.g., 5th index if 0-based) is a global attention layer
        x = layer.global_self_attn(x)            # full attention (standard)
    else:
        # Local attention layer
        x = layer.local_self_attn(x, window=1024)  # sliding window of 1024
    # ...then pass x through the MLP, etc., and continue
```

In practice, you can implement this by having two attention implementations in your layer class. For example, you might subclass `nn.Module` for a `TransformerBlock` that has `self.attn` as a normal `MultiheadAttention`. You can still use one `nn.MultiheadAttention` for both, by providing an `attn_mask` when you call it for local layers. Alternatively, you can write a custom attention `forward` using the function above for local layers. For clarity, you could also create two different module classes (`LocalAttentionLayer` and `GlobalAttentionLayer`) if that helps structure the code.

**Example:** Below is a simple conceptual example using PyTorch's `MultiheadAttention` with a sliding window mask:

```python
import torch
import torch.nn as nn
import math

def generate_sliding_mask(seq_len, window, device):
    """Generate a boolean mask of shape (seq_len, seq_len) for local attention."""
    idx = torch.arange(seq_len, device=device)
    mask = (idx[:, None] - idx[None, :] < window) & (idx[:, None] - idx[None, :] >= 0)
    # MultiheadAttention expects positions to mask *out* as True (it masks True positions)
    # So we need the inverted mask (False = allowed, True = disallowed)
    inv_mask = ~mask
    return inv_mask  # shape (seq_len, seq_len)

# Example transformer block with alternating attention
class LocalGlobalBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.window_size = window_size
        # (In practice, you'd also have feed-forward sublayer, norm, etc.)

    def forward(self, x, is_global: bool):
        # x shape: (batch, seq_len, embed_dim)
        if is_global:
            # Global attention: attend to all tokens (causal mask will be applied inside if needed)
            attn_out, _ = self.attn(x, x, x)  # (batch, seq_len, embed_dim)
        else:
            seq_len = x.size(1)
            mask = generate_sliding_mask(seq_len, self.window_size, x.device)
            attn_out, _ = self.attn(x, x, x, attn_mask=mask)  # apply sliding window mask
        return attn_out

# Usage:
seq_len = 2048
batch_size = 4
embed_dim = 768
num_heads = 8
window_size = 128
x = torch.rand(batch_size, seq_len, embed_dim)
block = LocalGlobalBlock(embed_dim, num_heads, window_size)

# Assume we alternate: first layer local, second layer global (just an example)
out_local = block(x, is_global=False)   # local attention layer
out_global = block(out_local, is_global=True)  # global attention layer
```

In this snippet, `generate_sliding_mask` creates a boolean mask where `True` means "mask out this position." For `MultiheadAttention`, we invert the allowed window to that form. The local layer call uses `attn_mask=mask` to restrict attention, and the global layer call just uses the full sequence. In a complete Transformer, you'd alternate these layers. For instance, Gemma 3 uses 5 local layers then 1 global layer repeatedly ([developers.googleblog.com](https://developers.googleblog.com)), whereas Command-A uses 3 local then 1 global ([cohere.com](https://cohere.com)). You can experiment with different ratios (the optimal may depend on your context length and task – more global layers might improve quality but at a compute cost).

**Trade-offs and debugging on M3:**
When you introduce local attention, ensure your causal masking is still correct in those layers. The mask generation above already prevents attending to future tokens, so it's compatible with causal LM training. It's good to test a small dummy sequence and verify that a token at position `i` cannot attend to position `i - (window_size+1)` or earlier (they should be zeroed out in attention weights). Another thing to watch is that global and local layers might behave slightly differently. For example, if you use certain positional embeddings, you may need to adjust them. In long contexts, you might decide to reset or use relative position embeddings for local windows. The cited models handle this by different RoPE scales or none on global layers, but if you're not pushing extreme lengths, you can likely ignore that and use the same positional encoding for all layers.

From a performance standpoint on an Apple M3: using local attention will significantly reduce compute for those layers, which is great for the GPU. A window of 128 or 1024 is much smaller than, say, a 8k context. Memory wise, it also helps – for local layers, the attention matrix is `[seq_len x window_size]` instead of `[seq_len x seq_len]`. This not only avoids the Metal buffer issue for very large `seq_len`, but also cuts gradient memory. Global layers are still O(\(N^2\)), so they could be the bottleneck if \(N\) is huge. The idea is that since they are infrequent, you might get away with a larger \(N\). For example, maybe your M3 Pro cannot do a single 16k×16k attention matrix due to memory, but if only 1 out of every 6 layers tries it, you could gradient-checkpoint that layer or find ways to manage it (since it's not every layer). If even that is too much, you might combine this strategy with chunked attention within the global layer (though that's getting quite complex).

**Debugging tips:**
When alternating attentions, it helps to print out or assert which layers are local vs global to ensure the pattern is as expected. Also, verify that the outputs have the right shape and no NaNs – masking errors can sometimes cause NaNs if implemented incorrectly (e.g. forgetting to mask out some logits in softmax). PyTorch's SDPA and `MultiheadAttention` take care of the masking in a stable way, so leverage those. On MPS, if you encounter an issue (like the ROCm bug note about NaNs with boolean masks in some versions - *[github.com](https://github.com) – a different backend but similar concept)*, ensure you test on smaller inputs first.

Finally, monitor memory usage when using large windows or global layers on M3. If you push context length high, the one or two global layers might still use a lot of memory. You might need to decrease batch size or use FP16 to accommodate. The local layers will alleviate some of the pressure, which is exactly why this design is used in state-of-the-art models ([developers.googleblog.com](https://developers.googleblog.com), [developers.googleblog.com](https://developers.googleblog.com)). With careful implementation, you'll gain the ability to train with longer sequences on Apple Silicon than a naive all-global Transformer would allow, and you'll still retain strong performance thanks to those periodic global attention layers that connect the "local" information together.

**Summary:**
Alternating local and global attention is a proven strategy to scale up context length efficiently. It provides huge memory and speed benefits by limiting most layers to a fixed attention span ([developers.googleblog.com](https://developers.googleblog.com)), and only occasionally performing full attention. The trade-off in modeling capacity is minor when the ratio is chosen well, as evidenced by minimal perplexity differences in models like Gemma 3 ([developers.googleblog.com](https://developers.googleblog.com)). Implement this by using a sliding window mask in your PyTorch attention layers for local layers, and keep the usual full attention for global layers. This will allow your "educational" model to handle longer inputs on the M3 Pro (MPS) backend more gracefully. Good luck, and enjoy experimenting with these optimizations on your Mac!

### Sources:

1.  **PyTorch 2.7 Documentation – `scaled_dot_product_attention`** (notes on FlashAttention vs fallback)
    *   [pytorch.org](https://pytorch.org)
2.  **Raksheka R., Optimizing PyTorch MPS Attention** – MPS backend memory issues with large sequences and chunking solution
    *   [medium.com](https://medium.com)
    *   [medium.com](https://medium.com)
3.  **PyTorch Forums – Discussion of SDPA backends and FlashAttention availability**
    *   [discuss.pytorch.org](https://discuss.pytorch.org)
4.  **Rohan Paul, Caching Strategies in LLMs** – Explanation of why KV caching is for inference, not used in training
    *   [rohan-paul.com](https://rohan-paul.com)
5.  **Michał Oleszak, Transformers KV Caching Explained** – KV caching speeds up autoregressive inference by reusing past keys/values
    *   [neptune.ai](https://neptune.ai)
6.  **Google Developers Blog – Gemma 3 model architecture** (5:1 local-global attention for long context and memory efficiency)
    *   [developers.googleblog.com](https://developers.googleblog.com)
    *   [developers.googleblog.com](https://developers.googleblog.com)
7.  **Cohere Technical Report – Command-A model** (3:1 interleaved sliding window vs full attention layers)
    *   [cohere.com](https://cohere.com)