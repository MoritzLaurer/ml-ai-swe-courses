# B2: Transformer Attention Formulas

Building on the vector similarity concepts from Module B1, we now dive into one of the most important formulas in modern machine learning: the **attention mechanism**. This is the mathematical heart of Transformer models, which power everything from GPT to BERT to ChatGPT.

## 1. The Problem That Attention Solves

Before jumping into the mathematics, let's understand *why* attention exists and what problem it elegantly solves.

### The Context Problem

Imagine you're reading this sentence: "The animal didn't cross the street because **it** was too tired."

As a human, you intuitively know that "it" refers to "the animal," not "the street." But how would you teach a computer to make this connection? The word "it" could potentially refer to any previous noun, but somehow we know which one is relevant.

**This is the attention problem:** Given a sequence of words (or any sequential data), how do we determine which previous elements are most relevant to understanding the current element?

### Traditional Approaches and Their Limitations

Before attention, neural networks processed sequences using:

*   **Recurrent Neural Networks (RNNs):** Process one word at a time, left to right, maintaining a "hidden state" that summarizes everything seen so far.
    *   **Problem:** By the time you reach "it," the information about "animal" might be diluted or forgotten, especially in long sequences.

*   **Convolutional approaches:** Use fixed-size windows to look at nearby words.
    *   **Problem:** Can only connect words within a limited window size.

### The Attention Solution

**Attention allows every word to directly "look at" and connect to every other word in the sequence.** When processing "it," the attention mechanism can directly examine "animal," "street," "cross," etc., and determine which ones are most relevant.

This is revolutionary because:
1. **No information loss:** Direct connections mean no forgetting
2. **Parallelizable:** Unlike RNNs, all connections can be computed simultaneously
3. **Interpretable:** We can visualize which words attend to which others

## 2. The Famous Scaled Dot-Product Attention Formula

Here's the formula that changed everything:

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

This looks intimidating at first, but it's actually quite elegant. Let's break it down piece by piece.

### 2.1 The Three Matrices: Q, K, V

The attention mechanism uses three different "views" of the input data:

*   **Q (Queries):** "What am I looking for?"
*   **K (Keys):** "What can I offer as context?"
*   **V (Values):** "What information do I actually contain?"

**Real-world Analogy:** Think of a library search system:
*   **Query (Q):** Your search terms ("Find books about machine learning")
*   **Keys (K):** The searchable metadata of each book (title, keywords, summary)
*   **Values (V):** The actual content of each book

The search system:
1. Compares your query against all the keys (book metadata)
2. Ranks how relevant each book is to your search
3. Returns the content (values) of the most relevant books

### 2.2 Step-by-Step Formula Breakdown

Let's trace through the formula from inside out:

#### Step 1: $QK^T$ (Computing Similarity Scores)

This is where our knowledge from Module B1 becomes crucial! 

*   $Q$ is an $n \times d_k$ matrix where each row is a query vector
*   $K$ is an $m \times d_k$ matrix where each row is a key vector  
*   $K^T$ is the transpose of $K$, making it $d_k \times m$
*   $QK^T$ results in an $n \times m$ matrix

**What $QK^T$ does:** Each element $(QK^T)_{ij}$ is the dot product between query $i$ and key $j$:
$$(QK^T)_{ij} = \mathbf{q}_i^T \mathbf{k}_j = \sum_{d=1}^{d_k} q_{id} \cdot k_{jd}$$

This is exactly the dot product formula from Module B1! Each dot product measures how much query $i$ "aligns with" or "is interested in" key $j$.

#### Step 2: $\frac{QK^T}{\sqrt{d_k}}$ (Scaling for Stability)

We divide all the dot products by $\sqrt{d_k}$ (the square root of the key dimension).

**Why scaling is necessary:** As the dimension $d_k$ increases, dot products tend to grow larger in magnitude. This can cause problems when we apply the softmax function (next step), because very large values can make the softmax outputs extremely close to 0 or 1, reducing the gradient flow during training.

**The mathematical intuition:** If elements of $\mathbf{q}$ and $\mathbf{k}$ are drawn from distributions with unit variance, their dot product will have variance proportional to $d_k$. Dividing by $\sqrt{d_k}$ normalizes this variance back to a reasonable scale.

#### Step 3: $\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$ (Converting to Probabilities)

The softmax function converts the scaled dot products into a probability distribution. For each query $i$, we get a probability distribution over all keys:

$$\text{softmax}(\mathbf{s})_j = \frac{\exp(s_j)}{\sum_{k=1}^{m} \exp(s_k)}$$

**What this achieves:**
*   All attention weights for each query sum to 1: $\sum_{j=1}^{m} \alpha_{ij} = 1$
*   Higher similarity scores get higher attention weights
*   The attention weights $\alpha_{ij}$ represent "how much should query $i$ pay attention to key $j$?"

#### Step 4: $\left[\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)\right]V$ (Weighted Combination)

Finally, we use these attention weights to create a weighted combination of the value vectors.

*   The softmax output is an $n \times m$ matrix of attention weights
*   $V$ is an $m \times d_v$ matrix where each row is a value vector
*   The final result is an $n \times d_v$ matrix

**What the final multiplication does:** For each query $i$, the output is:
$$\text{output}_i = \sum_{j=1}^{m} \alpha_{ij} \mathbf{v}_j$$

This is a weighted average of all value vectors, where the weights are determined by how much each query attends to each key.

## 3. Understanding Q, K, V: The Heart of Attention

### 3.1 Why Three Different Matrices?

You might wonder: "Why not just use the same vectors for queries, keys, and values?" The answer lies in the flexibility this provides:

**Separation of Concerns:**
*   **Keys and Queries** determine *what to attend to* (the similarity/relevance computation)
*   **Values** determine *what information to extract* (the actual content that gets mixed)

**Real-world Analogy:** Consider a restaurant recommendation system:
*   **Your query:** "I want spicy, affordable food" 
*   **Restaurant keys:** "Italian, expensive, mild" vs. "Mexican, cheap, very spicy"
*   **Restaurant values:** Full details including menu, reviews, location, etc.

The key helps determine relevance, but the value contains the actual information you get back.

### 3.2 How Q, K, V Are Created

In practice, Q, K, and V are created from the same input through linear transformations:

$$Q = XW_Q$$
$$K = XW_K$$  
$$V = XW_V$$

Where:
*   $X$ is the input matrix (e.g., word embeddings)
*   $W_Q, W_K, W_V$ are learned parameter matrices
*   These transformations project the input into different "subspaces" optimized for their specific roles

**Key Insight:** The model *learns* how to transform the input into effective queries, keys, and values during training. Different transformation matrices allow the model to use the same input information in three different ways.

## 4. The Scaling Factor $\sqrt{d_k}$: A Crucial Detail

The scaling factor $\sqrt{d_k}$ might seem like a minor detail, but it's actually crucial for the attention mechanism to work properly.

### 4.1 The Problem Without Scaling

Consider what happens as $d_k$ (the dimension of queries and keys) increases:

*   Dot products $\mathbf{q}^T\mathbf{k}$ grow in magnitude proportionally to $d_k$
*   Very large positive or negative values fed into softmax create extremely peaked probability distributions
*   This leads to attention weights that are very close to 0 or 1, with little in between

### 4.2 Mathematical Intuition

Assume the elements of $\mathbf{q}$ and $\mathbf{k}$ are independent random variables with mean 0 and variance 1. Then:

*   The dot product $\mathbf{q}^T\mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$ has:
    *   Mean: 0 (since each $q_i k_i$ has mean 0)
    *   Variance: $d_k$ (since we're summing $d_k$ independent terms each with variance 1)

*   Dividing by $\sqrt{d_k}$ gives us a scaled dot product with:
    *   Mean: 0
    *   Variance: 1

This keeps the variance constant regardless of the dimension $d_k$, ensuring stable softmax behavior.

### 4.3 Impact on Training

Without proper scaling:
*   **Gradient problems:** Extremely peaked softmax distributions have very small gradients
*   **Attention collapse:** The model might attend to only one position instead of learning nuanced attention patterns
*   **Training instability:** Large variations in attention behavior as dimensions change

The $\sqrt{d_k}$ scaling elegantly solves all these issues with a simple normalization.

## 5. Multi-Head Attention: Parallel Attention Mechanisms

The complete Transformer attention mechanism uses **multi-head attention**, which runs multiple attention mechanisms in parallel:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

where each head is:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### 5.1 Why Multiple Heads?

**Different types of relationships:** A single attention head might focus on one type of relationship (e.g., syntactic dependencies), while another head focuses on semantic relationships.

**Richer representations:** Multiple heads allow the model to attend to information from different representation subspaces simultaneously.

**Real-world Analogy:** When you read a sentence, you simultaneously pay attention to:
*   **Syntactic relationships:** Which words modify which others grammatically
*   **Semantic relationships:** Which words are related in meaning  
*   **Discourse relationships:** How the current sentence relates to previous context
*   **Pragmatic relationships:** Implied meanings and references

Multi-head attention allows the model to learn these different types of attention patterns in parallel.

### 5.2 The Complete Multi-Head Formula Breakdown

Let's unpack the multi-head formula:

#### Step 1: Parallel Head Computation
Each head $i$ computes attention using its own learned projections:
*   $QW_i^Q$: Projects queries into head $i$'s query subspace
*   $KW_i^K$: Projects keys into head $i$'s key subspace  
*   $VW_i^V$: Projects values into head $i$'s value subspace

#### Step 2: Concatenation
$$\text{Concat}(\text{head}_1, \ldots, \text{head}_h)$$

The outputs from all heads are concatenated along the feature dimension. If each head outputs $d_v$ dimensions and we have $h$ heads, the concatenated result has dimension $h \times d_v$.

#### Step 3: Final Linear Transformation
$$[\text{Concatenated heads}] \times W^O$$

A final linear transformation $W^O$ projects the concatenated heads back to the desired output dimension. This allows the heads to interact and combine their information.

### 5.3 Dimensional Analysis

Let's trace through the dimensions (assuming $d_{model} = 512$, $h = 8$ heads):

*   **Input:** $X \in \mathbb{R}^{n \times 512}$ (sequence length $n$, model dimension 512)
*   **Per-head projections:** Each $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{512 \times 64}$ (512 to 64 dimensions per head)
*   **Per-head attention output:** $\mathbb{R}^{n \times 64}$
*   **After concatenation:** $\mathbb{R}^{n \times 512}$ (8 heads × 64 dimensions each)
*   **Final output:** $W^O \in \mathbb{R}^{512 \times 512}$, so output is $\mathbb{R}^{n \times 512}$

**Key insight:** The total dimension is preserved ($h \times d_k = d_{model}$), but the computation is split across multiple specialized heads.

## 6. Matrix Dimensions and Data Flow

Understanding the exact matrix dimensions helps make the attention mechanism concrete. Let's trace through a specific example:

### 6.1 Example Setup
*   Sequence length: $n = 10$ (10 words)
*   Model dimension: $d_{model} = 512$
*   Number of heads: $h = 8$
*   Per-head dimension: $d_k = d_v = 64$ (so $h \times d_k = 8 \times 64 = 512$)

### 6.2 Step-by-Step Dimensional Analysis

**Input Embeddings:**
*   $X \in \mathbb{R}^{10 \times 512}$ (10 words, each with 512-dimensional embedding)

**Linear Projections (for one head):**
*   $W^Q \in \mathbb{R}^{512 \times 64}$, so $Q = XW^Q \in \mathbb{R}^{10 \times 64}$
*   $W^K \in \mathbb{R}^{512 \times 64}$, so $K = XW^K \in \mathbb{R}^{10 \times 64}$
*   $W^V \in \mathbb{R}^{512 \times 64}$, so $V = XW^V \in \mathbb{R}^{10 \times 64}$

**Attention Score Computation:**
*   $QK^T$: $(10 \times 64) \times (64 \times 10) = 10 \times 10$
*   Each element $(QK^T)_{ij}$ represents how much word $i$ attends to word $j$

**After Softmax:**
*   Still $10 \times 10$, but now each row sums to 1 (probability distribution)

**Final Output (for one head):**
*   $(\text{softmax weights}) \times V$: $(10 \times 10) \times (10 \times 64) = 10 \times 64$

**After Multi-Head Concatenation:**
*   8 heads × 64 dimensions = $10 \times 512$

### 6.3 The Self-Attention Case

In **self-attention** (the most common case), the queries, keys, and values all come from the same input sequence. This means:
*   Every word can attend to every other word (including itself)
*   The attention matrix is $n \times n$ where $n$ is the sequence length
*   Position $(i,j)$ in the attention matrix shows how much word $i$ attends to word $j$

**Visualization Insight:** You can visualize the attention matrix as a heatmap where darker colors indicate stronger attention. This often reveals meaningful linguistic patterns!

## 7. Common Variations and Extensions

### 7.1 Masked Attention

In language modeling (like GPT), we need **causal masking** to prevent the model from "cheating" by looking at future words:

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\text{mask}\left(\frac{QK^T}{\sqrt{d_k}}\right)\right)V$$

The mask sets attention scores to $-\infty$ for future positions, making their softmax probabilities 0.

### 7.2 Relative Position Encodings

Some models modify the attention to include relative position information:

$$\text{score}_{ij} = \frac{\mathbf{q}_i^T\mathbf{k}_j + \mathbf{q}_i^T\mathbf{r}_{i-j}}{\sqrt{d_k}}$$

where $\mathbf{r}_{i-j}$ is a learned embedding for the relative distance between positions $i$ and $j$.

### 7.3 Cross-Attention

In encoder-decoder architectures, **cross-attention** allows the decoder to attend to the encoder's outputs:
*   Queries come from the decoder
*   Keys and values come from the encoder
*   This enables the decoder to access the entire input sequence when generating each output

## 8. Reading Attention Formulas in Papers

### 8.1 Common Notation Variations

**Attention weights:**
*   $\alpha_{ij}$, $A_{ij}$, or $w_{ij}$ for attention from position $i$ to position $j$

**Different variable names:**
*   Sometimes you'll see $H$ instead of $V$ for "hidden states"
*   $\mathbf{s}_{ij}$ for raw attention scores before softmax
*   $\mathbf{c}_i$ for the final context vector for position $i$

**Temperature scaling:**
*   Some papers include a learnable temperature parameter: $\frac{QK^T}{\tau}$ where $\tau$ is learned

### 8.2 Efficiency Modifications

Modern papers often propose efficiency improvements:

**Sparse attention:**
*   Only compute attention for a subset of positions
*   Examples: local windows, strided patterns, random sampling

**Linear attention:**
*   Reformulate to avoid the quadratic $n^2$ cost
*   Often involves kernel methods or low-rank approximations

**Memory-efficient attention:**
*   Recompute attention during backward pass instead of storing large matrices
*   Critical for training on very long sequences

### 8.3 Architecture-Specific Variations

**BERT-style (bidirectional):**
*   All positions can attend to all other positions
*   Uses special tokens like [CLS] and [SEP]

**GPT-style (autoregressive):**
*   Causal masking prevents attending to future positions
*   Often includes special beginning-of-sequence tokens

**Encoder-Decoder:**
*   Self-attention in encoder and decoder
*   Cross-attention from decoder to encoder
*   Often includes different masking strategies

## Summary

The attention mechanism, encapsulated in the elegant formula:

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

represents one of the most important breakthroughs in machine learning. Understanding this formula deeply means understanding:

1. **The similarity computation** ($QK^T$): Using dot products from Module B1 to measure relevance
2. **The scaling factor** ($\sqrt{d_k}$): Ensuring numerical stability across different dimensions  
3. **The probability distribution** (softmax): Converting similarities to attention weights
4. **The information mixing** (multiplication by $V$): Creating weighted combinations of value vectors

Multi-head attention extends this by running multiple attention mechanisms in parallel, allowing the model to capture different types of relationships simultaneously.

These mathematical foundations power the language models that are transforming AI today. In our next module, we'll explore the other key components of Transformer architectures and see how attention fits into the bigger picture!