# Curriculum: Understanding Formulas in ML Research Papers

This curriculum is designed to help engineers and researchers understand mathematical formulas commonly found in ML research papers, with a focus on LLMs, loss functions, and reinforcement learning.

## Chapter A: Foundations of Mathematical Notation

### A1_introduction_to_mathematical_notation.md
- **Content:**
    - Basic mathematical symbols: summation ($\sum$), product ($\prod$), integrals ($\int$), derivatives ($\frac{d}{dx}$).
    - Common Greek letters used in ML (e.g., $\alpha, \beta, \theta, \phi, \sigma, \mu, \epsilon$).
    - Understanding sets, elements, and common set notation.
    - Reading and interpreting complex expressions with multiple operators and indices.
    - Brief introduction to LaTeX for recognizing formula structures.

### A2_linear_algebra_in_ml_formulas.md
- **Content:**
    - Notation for scalars, vectors (bold lowercase, e.g., $\mathbf{x}$), and matrices (bold uppercase, e.g., $\mathbf{W}$).
    - Representing operations: vector/matrix addition, scalar multiplication.
    - Dot product ($\mathbf{a} \cdot \mathbf{b}$ or $\mathbf{a}^T\mathbf{b}$) and its use in formulas.
    - Matrix multiplication ($\mathbf{AB}$) and understanding input/output dimensions.
    - Transposition ($\mathbf{W}^T$) and its meaning in formulas like $y = W^T x + b$.
    - Element-wise operations (Hadamard product $\odot$).

### A3_probability_and_statistics_in_ml_formulas.md
- **Content:**
    - Notation for probability: $P(A)$, $P(A|B)$ (conditional probability).
    - Random variables (uppercase, e.g., X) and their realizations (lowercase, e.g., x).
    - Expected value: $\mathbb{E}_{x \sim P}[f(x)]$ or $\mathbb{E}[X]$.
    - Variance: $\text{Var}(X)$.
    - Common probability distributions and their notation (e.g., $x \sim \mathcal{N}(\mu, \sigma^2)$ for Normal).
    - Argmax/argmin notation: $\text{argmax}_x f(x)$.
    - Notation for sampling: $x \sim D$ (x sampled from distribution D).

### A4_calculus_in_ml_formulas.md
- **Content:**
    - Derivatives: $\frac{df}{dx}$, partial derivatives: $\frac{\partial L}{\partial w_i}$.
    - Gradient notation: $\nabla J(\theta)$ or $\nabla_\theta J$.
    - Understanding gradients in the context of optimization (e.g., $\theta_{t+1} = \theta_t - \eta \nabla J(\theta_t)$).
    - Integrals in theoretical contexts (e.g., marginalization $\int p(x,y) dy$).
    - Subscripts and superscripts in derivative notation (e.g., with respect to specific variables).

## Chapter B: Decoding Formulas in Language Models

### B1_vector_representation_formulas.md
- **Content:**
    - Formulas involving vector embeddings.
    - Cosine similarity: $\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$.
    - Dot-product for measuring similarity or attention scores.
    - Python code examples using `numpy` or `torch` for selected operations (included as code blocks with explanations).

### B2_transformer_attention_formulas.md
- **Content:**
    - Scaled Dot-Product Attention formula: $Attention(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$.
    - Breakdown of Q (Query), K (Key), V (Value) matrices and $d_k$ (dimension of keys).
    - Multi-Head Attention: Concatenation and linear transformation formulas.
    - Visualizing the data flow and matrix operations.

### B3_common_llm_layer_formulas.md
- **Content:**
    - Feed-Forward Networks (FFNs) in Transformers: $FFN(x) = \text{max}(0, xW_1 + b_1)W_2 + b_2$ (using ReLU as an example).
    - Layer Normalization (LayerNorm) formula.
    - Residual Connections: $x_{out} = \text{LayerNorm}(x_{in} + \text{SubLayer}(x_{in}))$.
    - Parameter-Efficient Fine-tuning: LoRA formula $h = W_0 x + \Delta W x = W_0 x + BA x$ where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll d$.
    - Python code snippets illustrating the shapes and operations (included as code blocks with detailed explanations).

## Chapter C: Interpreting Loss Function Formulas

### C1_essential_loss_function_formulas.md
- **Content:**
    - General structure: $\mathcal{L}(\theta)$ or $J(\theta)$, often an average or sum over samples.
    - Mean Squared Error (MSE): $\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$.
    - Mean Absolute Error (MAE): $\mathcal{L}_{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$.
    - Binary Cross-Entropy (Log Loss): $\mathcal{L}_{BCE} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$.
    - Categorical Cross-Entropy: $\mathcal{L}_{CCE} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$.
    - Kullback-Leibler (KL) Divergence: $D_{KL}(P||Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$ for measuring distributional differences, commonly used in regularization and RLHF.
    - Python implementations demonstrating calculation for small examples (included as code blocks with step-by-step explanations).

### C2_autoregressive_loss_formulas_for_llms.md
- **Content:**
    - Next-token prediction objective for autoregressive LLMs.
    - Formula: $\mathcal{L} = - \sum_{t=1}^{T} \log P(x_t | x_{<t}; \theta)$ for a single sequence.
    - Explanation of conditional probability $P(x_t | x_{<t}; \theta)$ typically from a softmax output.
    - From Loss to Evaluation: Perplexity formula $\text{PPL} = \exp(-\frac{1}{N}\sum_{i=1}^N \log P(w_i|w_{<i}))$ and its connection to the autoregressive loss.
    - Python example calculating this loss for a small sequence and model logits (included as code blocks with detailed walkthrough).

## Chapter D: Key Formulas in Reinforcement Learning

### D1_foundational_rl_formulas.md
- **Content:**
    - Markov Decision Process (MDP) notation: States $S$, Actions $A$, Rewards $R(s,a,s')$, Transition Probabilities $P(s'|s,a)$, Discount factor $\gamma$.
    - Policy notation: $\pi(a|s)$ (stochastic) or $a = \mu(s)$ (deterministic).
    - State-Value function: $V^\pi(s) = \mathbb{E}_\pi [ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t=s ]$.
    - Action-Value function (Q-function): $Q^\pi(s,a) = \mathbb{E}_\pi [ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t=s, A_t=a ]$.
    - The Bellman Equation for $V^\pi(s)$ and $Q^\pi(s,a)$.
    - Bellman Optimality Equations for $V^*(s)$ and $Q^*(s,a)$.

### D2_policy_optimization_formulas.md
- **Content:**
    - The Policy Gradient Theorem: $\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [ (\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)) R(\tau) ]$. (One common form)
    - REINFORCE algorithm update rule.
    - Proximal Policy Optimization (PPO) clipped surrogate objective function:
      $$ L^{CLIP}(\theta) = \hat{\mathbb{E}}_t [ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) ] $$
      (with explanation of $r_t(\theta)$ and $\hat{A}_t$).
    - Python code examples illustrating these concepts (included as code blocks with explanations).

### D3_reward_modeling_formulas_in_rlhf.md
- **Content:**
    - Concept of a learned reward model $r_\phi(x,y)$ in RLHF, where $x$ is prompt and $y$ is completion.
    - Preference pair data: $(x, y_w, y_l)$ where $y_w$ is preferred over $y_l$.
    - Bradley-Terry model for preference probability: $P(y_w \succ y_l | x) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$.
    - Loss function for training the reward model:
      $$ \mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l) \sim D} [\log(\sigma(r_\phi(x, y_w) - r_\phi(x, y_l)))] $$
    - Alternative: Direct Preference Optimization (DPO) that skips reward modeling with formula:
      $$ \mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x,y_w,y_l) \sim D}[\log \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)})] $$
    - Python code snippet for the reward model loss calculation (included as code blocks with step-by-step explanation).

## Chapter E: Application and Practice

### E1_deconstructing_formulas_in_a_recent_paper.md
- **Content:**
    - Select a recent ML paper (e.g., related to LLMs or RL).
    - Identify 2-3 key or novel formulas.
    - Step-by-step breakdown of these formulas using the concepts learned in the curriculum.
    - Discussion on how the math translates to the paper's contributions and what each part of the formula signifies.

### E2_implementing_a_formula_from_scratch.md
- **Content:**
    - Choose a moderately complex formula not explicitly covered (or a variation of one covered).
    - Guide the learner through implementing it in Python with `numpy` or `torch`.
    - Emphasize mapping mathematical symbols and operations to code constructs.
    - Test with sample inputs.
    - All Python code included as code blocks with detailed explanations and comments.

---

**Note on Module Structure:**

All modules are **Markdown files (.md)**. These files contain:

*   **Mathematical explanations and derivations** using LaTeX notation (e.g., `$\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$`).
*   **Python code examples** included as fenced code blocks with detailed explanations and comments to help learners understand how mathematical formulas translate to code.
*   **Step-by-step breakdowns** of complex formulas to build understanding progressively.

Example of a block formula in Markdown:
 
$$
\mathcal{L}_{preference} = -\mathbb{E}_{(x, y_w, y_l) \sim D} [\log(\sigma(r_{\phi}(x, y_w) - r_{\phi}(x, y_l)))]
$$

VSCode with appropriate extensions (like "Markdown+Math") will render the LaTeX mathematics beautifully in the preview.

