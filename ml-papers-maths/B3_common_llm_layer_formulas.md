# B3: Common LLM Layer Formulas

Now that we understand attention mechanisms from Module B2, let's explore the other crucial components that make modern language models work. While attention gets most of the spotlight, these "supporting" components are equally important for creating stable, trainable, and powerful models.

## 1. The Complete Picture: Beyond Attention

A Transformer layer isn't just attention—it's a carefully orchestrated combination of several components:

1. **Multi-Head Attention** (covered in B2)
2. **Feed-Forward Networks (FFNs)** - point-wise processing and expansion
3. **Layer Normalization** - stabilizing training dynamics  
4. **Residual Connections** - enabling information flow through deep networks

Each component solves a specific problem, and together they create a robust architecture that can be trained effectively at scale.

**The Big Picture Formula:** A complete Transformer block can be written as:
$$\begin{align}
\mathbf{z}_1 &= \text{LayerNorm}(\mathbf{x} + \text{MultiHeadAttention}(\mathbf{x})) \\
\mathbf{z}_2 &= \text{LayerNorm}(\mathbf{z}_1 + \text{FFN}(\mathbf{z}_1))
\end{align}$$

Don't worry if this looks complex—we'll break down each piece!

## 2. Feed-Forward Networks (FFNs): The Computational Workhorses

### 2.1 What FFNs Do

While attention handles the "routing" of information (deciding what to attend to), FFNs do the actual "computation" on that information. Think of attention as the brain's executive function deciding what to focus on, and FFNs as the processing units that actually work with that focused information.

**The Standard FFN Formula:**
$$\text{FFN}(\mathbf{x}) = \text{activation}(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$

Most commonly with ReLU activation:
$$\text{FFN}(\mathbf{x}) = \max(0, \mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$

### 2.2 Breaking Down the FFN Formula

Let's trace through this step by step:

#### Step 1: First Linear Transformation
$$\mathbf{h}_1 = \mathbf{x}\mathbf{W}_1 + \mathbf{b}_1$$

**What happens:** 
- Input $\mathbf{x} \in \mathbb{R}^{d_{model}}$ (e.g., 768-dimensional)
- Weight matrix $\mathbf{W}_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$ (e.g., $768 \times 3072$)
- This **expands** the representation to a higher dimension $d_{ff}$ (typically $4 \times d_{model}$)

**Why expansion?** The larger intermediate dimension gives the network more "computational space" to work with. It's like having more scratch paper to work out a complex math problem.

#### Step 2: Non-linear Activation
$$\mathbf{h}_2 = \text{activation}(\mathbf{h}_1)$$

**With ReLU:** $\text{ReLU}(x) = \max(0, x)$ applied element-wise
- Introduces non-linearity (crucial for learning complex patterns)
- ReLU zeros out negative values, creating sparsity
- Other common choices: GELU, Swish, SiLU

**Why non-linearity matters:** Without activation functions, stacking linear layers would still be equivalent to a single linear layer. Non-linearity is what gives neural networks their power.

#### Step 3: Second Linear Transformation  
$$\text{FFN}(\mathbf{x}) = \mathbf{h}_2\mathbf{W}_2 + \mathbf{b}_2$$

**What happens:**
- Weight matrix $\mathbf{W}_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$ (e.g., $3072 \times 768$)
- This **compresses** back to the original dimension $d_{model}$
- The result can be added back to the input via residual connections

### 2.3 The Expand-and-Contract Pattern

The FFN follows an **expand-and-contract** pattern:
$$d_{model} \rightarrow d_{ff} \rightarrow d_{model}$$
$$768 \rightarrow 3072 \rightarrow 768$$

**Intuition:** This is like having a conversation where you:
1. **Expand:** Brainstorm many possible ideas and connections (wider representation)
2. **Filter:** Apply non-linear reasoning to select and combine ideas (activation)
3. **Contract:** Distill back to the essential insights (original dimension)

### 2.4 Why FFNs Are Essential

**Position-wise processing:** Unlike attention (which mixes information across positions), FFNs process each position independently. This provides:
- **Local computation:** Deep processing of individual positions
- **Parallelization:** All positions can be processed simultaneously
- **Computational power:** The expansion provides modeling capacity

**Memory and computation:** FFNs typically contain the majority of a Transformer's parameters. In a model like GPT-3:
- FFN parameters: ~67% of total parameters
- Attention parameters: ~33% of total parameters

## 3. Layer Normalization: Keeping Training Stable

### 3.1 The Layer Normalization Formula

$$\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sigma} + \beta$$

Where:
- $\mu = \frac{1}{d} \sum_{i=1}^{d} x_i$ (mean across the feature dimension)
- $\sigma = \sqrt{\frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2}$ (standard deviation across features)
- $\gamma, \beta \in \mathbb{R}^d$ are learned parameters
- $\odot$ denotes element-wise multiplication

### 3.2 Understanding the Components

#### Step 1: Compute Statistics
For each vector $\mathbf{x} \in \mathbb{R}^d$, compute the mean and standard deviation across its features:

$$\mu = \frac{1}{d}(x_1 + x_2 + \cdots + x_d)$$
$$\sigma = \sqrt{\frac{1}{d}((x_1-\mu)^2 + (x_2-\mu)^2 + \cdots + (x_d-\mu)^2)}$$

#### Step 2: Normalize
$$\hat{\mathbf{x}} = \frac{\mathbf{x} - \mu}{\sigma}$$

This centers the vector at zero mean and scales it to unit variance. Each feature now has:
- Mean: 0
- Standard deviation: 1

#### Step 3: Scale and Shift
$$\text{LayerNorm}(\mathbf{x}) = \gamma \odot \hat{\mathbf{x}} + \beta$$

The learned parameters $\gamma$ and $\beta$ allow the model to:
- **Scale:** $\gamma$ controls the variance of each feature
- **Shift:** $\beta$ controls the mean of each feature
- **Recover:** If needed, the model can learn $\gamma = \sigma, \beta = \mu$ to recover the original distribution

### 3.3 Why Layer Normalization Works

**Training stability:** Deep networks suffer from **internal covariate shift**—the distribution of inputs to each layer keeps changing during training, making learning difficult.

**Gradient flow:** LayerNorm helps gradients flow better through deep networks by:
- Preventing activation values from becoming too large or small
- Reducing the dependence on initialization
- Making the optimization landscape smoother

**Real-world Analogy:** Imagine you're a teacher, and every day your students come from completely different educational backgrounds. It would be hard to teach effectively! LayerNorm is like giving everyone a standardized prep course before each lesson, ensuring they all start from a similar baseline.

### 3.4 LayerNorm vs. BatchNorm

**BatchNorm** normalizes across the batch dimension (across different examples).
**LayerNorm** normalizes across the feature dimension (within each example).

For language models, LayerNorm is preferred because:
- **Sequence lengths vary:** Batch normalization struggles with variable-length sequences
- **Generation:** During text generation, you often process one token at a time (batch size = 1), making batch statistics meaningless
- **Consistency:** LayerNorm provides the same normalization whether training or generating

## 4. Residual Connections: Enabling Deep Networks

### 4.1 The Residual Connection Formula

Instead of computing $\mathbf{y} = F(\mathbf{x})$, residual connections compute:
$$\mathbf{y} = \mathbf{x} + F(\mathbf{x})$$

Where $F(\mathbf{x})$ represents any sub-layer (attention, FFN, etc.).

**In Transformers, this appears as:**
$$\begin{align}
\mathbf{z}_1 &= \mathbf{x} + \text{MultiHeadAttention}(\mathbf{x}) \\
\mathbf{z}_2 &= \mathbf{z}_1 + \text{FFN}(\mathbf{z}_1)
\end{align}$$

### 4.2 Why Residual Connections Are Crucial

#### Problem: Vanishing Gradients
In deep networks, gradients can become exponentially small as they backpropagate through many layers. This makes it nearly impossible to train very deep networks.

**Mathematical insight:** When computing gradients via the chain rule:
$$\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{x}}$$

Without residual connections: $\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \frac{\partial F(\mathbf{x})}{\partial \mathbf{x}}$

With residual connections: $\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \frac{\partial}{\partial \mathbf{x}}[\mathbf{x} + F(\mathbf{x})] = \mathbf{I} + \frac{\partial F(\mathbf{x})}{\partial \mathbf{x}}$

**The key insight:** The identity matrix $\mathbf{I}$ provides a "gradient highway" that always allows gradients to flow backward, regardless of what happens in $F(\mathbf{x})$.

#### Problem: Training Very Deep Networks
Before residual connections, training networks deeper than ~20 layers was extremely difficult. With residuals, we can train networks hundreds of layers deep.

**Intuition:** Residual connections provide an "easier path" for the network to learn. Instead of having to learn a complex function $F(\mathbf{x})$ from scratch, the network only needs to learn the **residual** (difference) from the identity mapping.

### 4.3 The Highway Metaphor

Think of residual connections as highways with on-ramps and off-ramps:

- **The highway (residual path):** $\mathbf{x}$ flows directly through with minimal processing
- **The on-ramp (sub-layer):** $F(\mathbf{x})$ processes the information and adds it back
- **Multiple lanes:** Information can flow through both paths simultaneously

This design allows:
- **Fast information flow:** Critical information can bypass complex processing when needed
- **Optional processing:** Each layer can choose how much to modify the representation
- **Graceful degradation:** If a layer learns poorly, it can approximate the identity function

## 5. Putting It All Together: The Complete Transformer Block

### 5.1 The Standard Transformer Layer Formula

Combining all components, a complete Transformer layer performs:

$$\begin{align}
\mathbf{z}_1 &= \text{LayerNorm}(\mathbf{x} + \text{MultiHeadAttention}(\mathbf{x})) \\
\mathbf{z}_2 &= \text{LayerNorm}(\mathbf{z}_1 + \text{FFN}(\mathbf{z}_1))
\end{align}$$

### 5.2 Step-by-Step Data Flow

Let's trace through what happens to a sequence of embeddings:

#### Input
- $\mathbf{X} \in \mathbb{R}^{n \times d_{model}}$ (sequence of $n$ tokens, each with $d_{model}$ dimensions)

#### Step 1: Self-Attention + Residual + LayerNorm
1. **Attention:** $\mathbf{A} = \text{MultiHeadAttention}(\mathbf{X})$ 
2. **Residual:** $\mathbf{R}_1 = \mathbf{X} + \mathbf{A}$
3. **LayerNorm:** $\mathbf{Z}_1 = \text{LayerNorm}(\mathbf{R}_1)$

#### Step 2: FFN + Residual + LayerNorm  
1. **FFN:** $\mathbf{F} = \text{FFN}(\mathbf{Z}_1)$
2. **Residual:** $\mathbf{R}_2 = \mathbf{Z}_1 + \mathbf{F}$
3. **LayerNorm:** $\mathbf{Z}_2 = \text{LayerNorm}(\mathbf{R}_2)$

#### Output
- $\mathbf{Z}_2 \in \mathbb{R}^{n \times d_{model}}$ (same shape as input, ready for the next layer)

### 5.3 Pre-Norm vs. Post-Norm

**Post-Norm (shown above):** LayerNorm applied after the residual connection
**Pre-Norm:** LayerNorm applied before the sub-layer

$$\begin{align}
\text{Pre-Norm: } \mathbf{z}_1 &= \mathbf{x} + \text{MultiHeadAttention}(\text{LayerNorm}(\mathbf{x})) \\
\mathbf{z}_2 &= \mathbf{z}_1 + \text{FFN}(\text{LayerNorm}(\mathbf{z}_1))
\end{align}$$

**Why Pre-Norm is often preferred:**
- Better gradient flow in very deep networks
- More stable training dynamics
- Used in many modern LLMs (GPT-3, PaLM, etc.)

## 6. LoRA: Parameter-Efficient Fine-Tuning

### 6.1 The LoRA Formula

Low-Rank Adaptation (LoRA) modifies the standard linear transformation:

**Standard:** $\mathbf{h} = \mathbf{x}\mathbf{W}$
**LoRA:** $\mathbf{h} = \mathbf{x}\mathbf{W}_0 + \mathbf{x}\Delta\mathbf{W} = \mathbf{x}\mathbf{W}_0 + \mathbf{x}\mathbf{B}\mathbf{A}$

Where:
- $\mathbf{W}_0$: Original pre-trained weights (frozen)
- $\Delta\mathbf{W} = \mathbf{B}\mathbf{A}$: Low-rank decomposition of the weight update
- $\mathbf{B} \in \mathbb{R}^{d \times r}$, $\mathbf{A} \in \mathbb{R}^{r \times k}$ with $r \ll d, k$

### 6.2 Understanding the Low-Rank Decomposition

#### The Key Insight
Instead of learning a full $d \times k$ matrix of updates $\Delta\mathbf{W}$, LoRA learns two smaller matrices that multiply to approximate it:

$$\Delta\mathbf{W}_{d \times k} \approx \mathbf{B}_{d \times r} \times \mathbf{A}_{r \times k}$$

#### Parameter Count Comparison
**Full fine-tuning:** Need to update all $d \times k$ parameters
**LoRA:** Only need to learn $d \times r + r \times k = r(d + k)$ parameters

**Example:** For a layer with $d = k = 4096$ and $r = 16$:
- Full: $4096 \times 4096 = 16.7M$ parameters
- LoRA: $16 \times (4096 + 4096) = 131K$ parameters
- **Reduction:** 128× fewer parameters!

### 6.3 Why LoRA Works

#### Hypothesis: Intrinsic Dimensionality
The key insight is that **most adaptation happens in a low-dimensional subspace**. When fine-tuning a pre-trained model for a specific task, you don't need to modify the entire weight space—most of the adaptation can be captured by changes in a much smaller subspace.

#### Mathematical Intuition
Think of the weight matrix $\mathbf{W}_0$ as defining a "basis" for the model's current capabilities. The LoRA adaptation $\mathbf{B}\mathbf{A}$ learns a low-rank "correction" that adjusts these capabilities for the new task.

**Analogy:** Imagine you're a skilled musician learning a new piece. You don't need to relearn how to play your instrument from scratch—you just need to learn the specific patterns and adjustments for this new piece. LoRA works similarly.

### 6.4 LoRA in Practice

#### Where to Apply LoRA
LoRA can be applied to any linear layer, but most commonly:
- **Query and Value projections** in attention: $\mathbf{W}_Q, \mathbf{W}_V$
- **Output projection** of attention: $\mathbf{W}_O$
- **FFN layers:** $\mathbf{W}_1, \mathbf{W}_2$

#### Initialization Strategy
- $\mathbf{A}$: Random Gaussian initialization
- $\mathbf{B}$: Zero initialization
- This ensures $\Delta\mathbf{W} = \mathbf{B}\mathbf{A} = \mathbf{0}$ at start, so the model begins identical to the pre-trained version

#### Scaling Factor
The complete LoRA formula often includes a scaling factor:
$$\mathbf{h} = \mathbf{x}\mathbf{W}_0 + \frac{\alpha}{r}\mathbf{x}\mathbf{B}\mathbf{A}$$

Where $\alpha$ is a hyperparameter that controls the magnitude of the adaptation.

## 7. Reading These Formulas in Papers

### 7.1 Common Notation Variations

**Feed-Forward Networks:**
- $\text{FFN}(\mathbf{x})$, $\text{MLP}(\mathbf{x})$, or $\text{FC}(\mathbf{x})$ (fully connected)
- Sometimes written as: $\mathbf{W}_2 \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$
- Activation functions: ReLU, GELU, Swish, SiLU

**Layer Normalization:**
- $\text{LN}(\mathbf{x})$, $\text{LayerNorm}(\mathbf{x})$, or $\text{Norm}(\mathbf{x})$
- Sometimes the $\gamma$ and $\beta$ parameters are omitted from the formula for brevity

**Residual Connections:**
- Often implicit in the notation
- Sometimes written as $\mathbf{y} = \mathbf{x} + \mathcal{F}(\mathbf{x})$ where $\mathcal{F}$ is any sub-layer

### 7.2 Architecture Variations

**RMSNorm (Root Mean Square Layer Normalization):**
$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}} \odot \gamma$$

Simplifies LayerNorm by removing the mean centering step.

**SwiGLU Activation:**
$$\text{SwiGLU}(\mathbf{x}) = \text{Swish}(\mathbf{x}\mathbf{W}_1) \odot (\mathbf{x}\mathbf{W}_2)$$

Uses two parallel linear transformations with element-wise multiplication.

**DeepNorm:**
$$\mathbf{y} = \text{LayerNorm}(\alpha \mathbf{x} + \mathbf{F}(\mathbf{x}))$$

Scales the residual connection to improve training stability in very deep networks.

### 7.3 Modern Architectural Trends

**Parallel Attention and FFN:**
Some recent architectures compute attention and FFN in parallel rather than sequentially:
$$\mathbf{y} = \mathbf{x} + \text{Attention}(\mathbf{x}) + \text{FFN}(\mathbf{x})$$

**Mixture of Experts (MoE):**
Replace the standard FFN with multiple expert networks:
$$\text{MoE}(\mathbf{x}) = \sum_{i=1}^{N} G(\mathbf{x})_i \cdot E_i(\mathbf{x})$$

Where $G(\mathbf{x})$ is a gating function and $E_i$ are expert networks.

## Summary

The complete Transformer architecture elegantly combines several key components:

1. **Multi-Head Attention:** Routes information based on content similarity
2. **Feed-Forward Networks:** Provide computational power through expand-and-contract processing  
3. **Layer Normalization:** Stabilizes training by normalizing activations
4. **Residual Connections:** Enable gradient flow in deep networks

The complete block formula:
$$\begin{align}
\mathbf{z}_1 &= \text{LayerNorm}(\mathbf{x} + \text{MultiHeadAttention}(\mathbf{x})) \\
\mathbf{z}_2 &= \text{LayerNorm}(\mathbf{z}_1 + \text{FFN}(\mathbf{z}_1))
\end{align}$$

Modern innovations like **LoRA** show how we can efficiently adapt these powerful architectures:
$$\mathbf{h} = \mathbf{x}\mathbf{W}_0 + \mathbf{x}\mathbf{B}\mathbf{A}$$

Understanding these formulas deeply prepares you to read and understand the vast majority of modern language model architectures. In our next chapter, we'll explore how these models are trained using various loss functions!