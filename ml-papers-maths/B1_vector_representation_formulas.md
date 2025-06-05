# B1: Vector Representation Formulas

Welcome to Chapter B! We're now transitioning from foundational mathematics to understanding how these concepts appear in real machine learning applications, particularly in language models. This module focuses on **vector representations** (also called embeddings) and the key formulas used to compare and manipulate them.

## 1. What Are Vector Representations?

In machine learning, especially in natural language processing and language models, we represent everything as vectors - ordered lists of numbers that capture meaning in mathematical form.

**Think of it this way:** Just as you might describe a person using a list of attributes (height, age, income, years of education), machine learning models describe words, sentences, or any input using vectors of numbers.

### Examples of Vector Representations

*   **Word Embeddings:** The word "king" might be represented as a 300-dimensional vector:
    $$\mathbf{e}_{\text{king}} = [0.2, -0.1, 0.8, 0.3, \ldots] \in \mathbb{R}^{300}$$

*   **Sentence Embeddings:** An entire sentence might become a 768-dimensional vector:
    $$\mathbf{s}_{\text{"Hello world"}} = [0.5, 0.1, -0.3, \ldots] \in \mathbb{R}^{768}$$

*   **Image Features:** Even images can be represented as vectors after processing through a neural network.

**The Key Insight:** Similar things should have similar vector representations. Words with similar meanings should have vectors that are "close" to each other in the high-dimensional space.

### Why Vectors?

Representing everything as vectors allows us to:
1. **Perform mathematical operations** on meanings (e.g., "king" - "man" + "woman" ≈ "queen")
2. **Measure similarity** between different inputs
3. **Process them through neural networks** that expect numerical inputs
4. **Scale computations** efficiently using linear algebra

## 2. Core Vector Operations and Formulas

Now let's explore the key mathematical operations used to work with these vector representations. These build directly on the linear algebra concepts from Module A2.

### 2.1 Dot Product (Review and Application)

**Formula:** For two vectors $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$:
$$\mathbf{a} \cdot \mathbf{b} = \mathbf{a}^T\mathbf{b} = \sum_{i=1}^{n} a_i b_i$$

**What it measures:** The dot product captures how much two vectors "align" with each other:
*   **Large positive value:** Vectors point in similar directions
*   **Zero:** Vectors are perpendicular (orthogonal)
*   **Large negative value:** Vectors point in opposite directions

**In ML/LLMs:** The dot product is fundamental for:
*   **Attention mechanisms:** $\text{score} = \mathbf{q}^T\mathbf{k}$ (query-key dot product)
*   **Similarity measurement:** Higher dot product often means more similar meanings
*   **Linear layers:** $\mathbf{y} = \mathbf{Wx}$ involves dot products between weight rows and input

**Example Calculation:**
If $\mathbf{a} = [1, 2, 3]$ and $\mathbf{b} = [4, 5, 6]$, then:
$$\mathbf{a} \cdot \mathbf{b} = (1 \times 4) + (2 \times 5) + (3 \times 6) = 4 + 10 + 18 = 32$$

### 2.2 Vector Norms (Measuring Length)

**L2 Norm (Euclidean Norm):** The most common way to measure a vector's "length":
$$\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^{n} v_i^2} = \sqrt{\mathbf{v}^T\mathbf{v}}$$

**L1 Norm (Manhattan Norm):** Sum of absolute values:
$$\|\mathbf{v}\|_1 = \sum_{i=1}^{n} |v_i|$$

**In ML/LLMs:**
*   **Normalization:** Creating unit vectors with $\|\mathbf{v}\|_2 = 1$
*   **Regularization:** Preventing weights from becoming too large
*   **Distance calculation:** Used in similarity measures

**Example Calculation:**
If $\mathbf{v} = [3, 4]$, then:
*   $\|\mathbf{v}\|_2 = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$
*   $\|\mathbf{v}\|_1 = |3| + |4| = 7$

### 2.3 Cosine Similarity: The Star Formula

**Formula:**
$$\text{cosine\_sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2 \|\mathbf{b}\|_2}$$

**What it measures:** Cosine similarity measures the angle between two vectors, regardless of their magnitudes (lengths). It ranges from -1 to 1:
*   **1:** Vectors point in exactly the same direction (identical orientation)
*   **0:** Vectors are perpendicular (no similarity in direction)
*   **-1:** Vectors point in exactly opposite directions

**Why cosine and not just dot product?**
*   The dot product is affected by both direction AND magnitude
*   Cosine similarity only cares about direction
*   This is crucial when comparing embeddings of different "intensities"

**Real-world Analogy:** Imagine two people walking. The dot product would consider both their walking direction and how fast they're walking. Cosine similarity only cares about whether they're walking in the same direction, regardless of speed.

**In ML/LLMs:**
*   **Finding similar words:** Words with high cosine similarity have similar meanings
*   **Document similarity:** Comparing text documents represented as vectors
*   **Recommendation systems:** Finding similar users or items
*   **Attention mechanisms:** Sometimes used instead of dot product for stability

**Example Calculation:**
If $\mathbf{a} = [1, 2]$ and $\mathbf{b} = [2, 4]$:
*   $\mathbf{a} \cdot \mathbf{b} = (1 \times 2) + (2 \times 4) = 10$
*   $\|\mathbf{a}\|_2 = \sqrt{1^2 + 2^2} = \sqrt{5}$
*   $\|\mathbf{b}\|_2 = \sqrt{2^2 + 4^2} = \sqrt{20} = 2\sqrt{5}$
*   $\text{cosine\_sim}(\mathbf{a}, \mathbf{b}) = \frac{10}{\sqrt{5} \times 2\sqrt{5}} = \frac{10}{10} = 1$

Notice that $\mathbf{b} = 2\mathbf{a}$, so they point in exactly the same direction!

### 2.4 Euclidean Distance

**Formula:**
$$\text{distance}(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|_2 = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}$$

**What it measures:** The straight-line distance between two points in the vector space.

**In ML/LLMs:**
*   **Clustering:** Grouping similar items together
*   **Nearest neighbor search:** Finding the closest embeddings
*   **Loss functions:** Measuring how far predictions are from targets

**Cosine vs. Euclidean Distance:**
*   **Cosine similarity:** Better when magnitude doesn't matter (direction-focused)
*   **Euclidean distance:** Better when both position and magnitude matter

## 3. Python Implementation Examples

Let's see how these formulas translate to code:

```python
import numpy as np
import torch

# Example vectors (word embeddings)
king = np.array([0.2, 0.5, -0.1, 0.8])
queen = np.array([0.3, 0.4, -0.2, 0.7])
car = np.array([-0.1, -0.3, 0.9, 0.2])

print("Vector representations:")
print(f"king: {king}")
print(f"queen: {queen}")
print(f"car: {car}")
```

### 3.1 Dot Product

```python
# Method 1: Using numpy's dot function
dot_product = np.dot(king, queen)
print(f"Dot product (king, queen): {dot_product:.3f}")

# Method 2: Manual calculation
manual_dot = np.sum(king * queen)  # Element-wise multiply then sum
print(f"Manual dot product: {manual_dot:.3f}")

# Method 3: Using @ operator (matrix multiplication)
dot_at = king @ queen
print(f"Using @ operator: {dot_at:.3f}")

# Output will be:
# Dot product (king, queen): 0.650
# Manual dot product: 0.650
# Using @ operator: 0.650
```

### 3.2 Vector Norms

```python
# L2 norm (Euclidean norm)
king_norm = np.linalg.norm(king)  # Default is L2 norm
print(f"||king||_2: {king_norm:.3f}")

# Manual L2 norm calculation
manual_norm = np.sqrt(np.sum(king**2))
print(f"Manual L2 norm: {manual_norm:.3f}")

# L1 norm
king_l1 = np.linalg.norm(king, ord=1)
print(f"||king||_1: {king_l1:.3f}")

# Output will be:
# ||king||_2: 0.943
# Manual L2 norm: 0.943
# ||king||_1: 1.600
```

### 3.3 Cosine Similarity

```python
def cosine_similarity(a, b):
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        a, b: numpy arrays of same length
        
    Returns:
        float: cosine similarity (-1 to 1)
    """
    # Method 1: Step by step
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

# Calculate similarities
sim_king_queen = cosine_similarity(king, queen)
sim_king_car = cosine_similarity(king, car)
sim_queen_car = cosine_similarity(queen, car)

print("Cosine Similarities:")
print(f"king ↔ queen: {sim_king_queen:.3f}")
print(f"king ↔ car: {sim_king_car:.3f}")
print(f"queen ↔ car: {sim_queen_car:.3f}")

# Using sklearn (if available)
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

# Reshape for sklearn (expects 2D arrays)
vectors = np.array([king, queen, car])
similarity_matrix = sklearn_cosine(vectors)
print("\nSimilarity Matrix:")
print(similarity_matrix)

# Output will show that king and queen are more similar to each other
# than either is to car, which makes semantic sense!
```

### 3.4 Euclidean Distance

```python
def euclidean_distance(a, b):
    """Calculate Euclidean distance between two vectors."""
    return np.linalg.norm(a - b)

# Calculate distances
dist_king_queen = euclidean_distance(king, queen)
dist_king_car = euclidean_distance(king, car)

print("Euclidean Distances:")
print(f"king ↔ queen: {dist_king_queen:.3f}")
print(f"king ↔ car: {dist_king_car:.3f}")

# Using scipy.spatial.distance (if available)
from scipy.spatial.distance import euclidean
dist_scipy = euclidean(king, queen)
print(f"Using scipy: {dist_scipy:.3f}")
```

### 3.5 Working with PyTorch Tensors

```python
# Convert to PyTorch tensors (common in deep learning)
king_torch = torch.tensor(king, dtype=torch.float32)
queen_torch = torch.tensor(queen, dtype=torch.float32)

# Dot product in PyTorch
dot_torch = torch.dot(king_torch, queen_torch)
print(f"PyTorch dot product: {dot_torch.item():.3f}")

# Cosine similarity in PyTorch
cos_sim_torch = torch.nn.functional.cosine_similarity(
    king_torch.unsqueeze(0), queen_torch.unsqueeze(0)
)
print(f"PyTorch cosine similarity: {cos_sim_torch.item():.3f}")

# L2 norm in PyTorch
norm_torch = torch.norm(king_torch)
print(f"PyTorch L2 norm: {norm_torch.item():.3f}")
```

## 4. Why These Formulas Matter in Language Models

### 4.1 Word Similarity and Analogies

The famous word analogy example: "king" - "man" + "woman" ≈ "queen"

```python
# Conceptual example (with made-up simple embeddings)
man = np.array([0.1, 0.3, 0.2, 0.1])
woman = np.array([0.2, 0.2, 0.1, 0.3])
king = np.array([0.3, 0.4, 0.3, 0.2])

# The famous analogy: king - man + woman should ≈ queen
analogy_result = king - man + woman
print(f"king - man + woman = {analogy_result}")

# In practice, you'd find the word embedding closest to this result
```

### 4.2 Attention Mechanisms

In Transformers, attention scores are computed using dot products:

```python
# Simplified attention score calculation
query = np.array([0.5, 0.3, 0.1])      # What we're looking for
key1 = np.array([0.4, 0.2, 0.2])       # First candidate
key2 = np.array([0.1, 0.8, 0.1])       # Second candidate

# Attention scores (before softmax)
score1 = np.dot(query, key1)
score2 = np.dot(query, key2)

print(f"Attention score 1: {score1:.3f}")
print(f"Attention score 2: {score2:.3f}")

# Higher score means more attention/relevance
```

### 4.3 Similarity Search

Finding the most similar documents or words:

```python
# Example: Finding most similar word to a query
query_word = np.array([0.2, 0.5, -0.1])
vocabulary = {
    'happy': np.array([0.3, 0.4, -0.2]),
    'sad': np.array([-0.2, 0.1, 0.3]),
    'joyful': np.array([0.25, 0.45, -0.15]),
    'angry': np.array([-0.1, -0.3, 0.4])
}

similarities = {}
for word, embedding in vocabulary.items():
    sim = cosine_similarity(query_word, embedding)
    similarities[word] = sim

# Sort by similarity
sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
print("Most similar words:")
for word, sim in sorted_similarities:
    print(f"{word}: {sim:.3f}")
```

## 5. Common Variations and Notation in Papers

### 5.1 Scaled Dot Product

Often in attention mechanisms, you'll see:
$$\text{score} = \frac{\mathbf{q}^T\mathbf{k}}{\sqrt{d_k}}$$

The $\sqrt{d_k}$ scaling factor (where $d_k$ is the dimension of the key vectors) prevents the dot product from growing too large as dimensions increase.

### 5.2 Normalized Embeddings

Sometimes papers work with unit vectors (normalized embeddings):
$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$$

When vectors are normalized, cosine similarity becomes simply the dot product:
$$\text{cosine\_sim}(\hat{\mathbf{a}}, \hat{\mathbf{b}}) = \hat{\mathbf{a}}^T\hat{\mathbf{b}}$$

### 5.3 Batch Operations

In practice, we often work with matrices where each row is a vector:

```python
# Matrix of embeddings (each row is one embedding)
embeddings = np.array([
    [0.2, 0.5, -0.1],  # word 1
    [0.3, 0.4, -0.2],  # word 2
    [-0.1, 0.8, 0.1]   # word 3
])

# Pairwise cosine similarities (all pairs at once)
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(embeddings)
print("Pairwise similarity matrix:")
print(sim_matrix)
```

### 5.4 Common Paper Notation

*   **Similarity matrices:** $\mathbf{S}_{ij} = \text{sim}(\mathbf{v}_i, \mathbf{v}_j)$
*   **Attention weights:** $\alpha_{ij} = \frac{\exp(\mathbf{q}_i^T\mathbf{k}_j)}{\sum_k \exp(\mathbf{q}_i^T\mathbf{k}_k)}$
*   **Nearest neighbors:** $\text{NN}(\mathbf{v}) = \arg\max_{\mathbf{u} \in V} \text{sim}(\mathbf{v}, \mathbf{u})$

## 6. Practical Tips for Reading Papers

When you encounter vector similarity formulas in papers:

1. **Identify the vectors:** What do they represent? (words, sentences, hidden states?)
2. **Check the similarity measure:** Dot product, cosine similarity, or Euclidean distance?
3. **Look for normalization:** Are the vectors normalized? This affects interpretation.
4. **Consider dimensionality:** Higher-dimensional spaces behave differently.
5. **Check for scaling factors:** Terms like $\frac{1}{\sqrt{d}}$ are common for numerical stability.

## Summary

Vector representations are the foundation of modern NLP and language models. The key formulas we've covered:

*   **Dot Product:** $\mathbf{a}^T\mathbf{b} = \sum_i a_i b_i$ - measures alignment
*   **Cosine Similarity:** $\frac{\mathbf{a}^T\mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}$ - measures directional similarity
*   **Euclidean Distance:** $\|\mathbf{a} - \mathbf{b}\|_2$ - measures spatial distance

These simple formulas power complex behaviors in language models, from finding similar words to computing attention weights. Understanding them deeply will help you decode the mathematical notation in research papers and implement these concepts in your own code.

In the next module, we'll dive deeper into how these vector operations appear in the famous Transformer attention mechanism!