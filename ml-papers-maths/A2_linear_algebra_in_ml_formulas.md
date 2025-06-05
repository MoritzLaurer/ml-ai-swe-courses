# A2: Linear Algebra in ML Formulas

Linear algebra is a branch of mathematics that deals with vector spaces and linear mappings between them. In machine learning, it's absolutely essential. We use it to represent our data, define our models, and describe the operations that transform data into predictions. This module will introduce the common notation for scalars, vectors, matrices, and the fundamental operations performed on them in ML contexts.

## 1. Scalars, Vectors, and Matrices: The Building Blocks

### Scalars
*   **What it is:** A scalar is simply a single number.
*   **Notation:** Typically denoted by lowercase italic letters (e.g., $s, a, x, k$). Greek letters are also commonly used for scalars (e.g., $\alpha, \beta, \lambda, \eta$ from Module A1).
*   **Examples in ML:**
    *   Learning rate: $\eta = 0.01$
    *   Regularization strength: $\lambda = 0.5$
    *   A single feature value: $x_1 = 3.7$ (where $x_1$ is the first feature of a data point)
    *   A bias term in a simple model: $b$

### Vectors
*   **What it is:** A vector is an ordered list of numbers. Geometrically, a vector can represent a point in space or a direction and magnitude. In ML, vectors are commonly used to represent data points (features) or parameters of a model.
*   **Notation:**
    *   **Bold lowercase letters:** This is the most common convention in ML literature (e.g., $\mathbf{x}, \mathbf{w}, \mathbf{v}, \mathbf{b}$).
    *   **Elements:** The individual numbers within a vector are its elements or components. If $\mathbf{x}$ is a vector, its $i$-th element is typically denoted by $x_i$ (lowercase, italic, not bold, with a subscript).
    *   **Dimension:** Vectors are often described as belonging to an $n$-dimensional real space, denoted $\mathbb{R}^n$. This means the vector has $n$ elements, and each element is a real number. For example, $\mathbf{x} \in \mathbb{R}^3$ means $\mathbf{x} = (x_1, x_2, x_3)$ or $\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}$.
    *   **Column vs. Row Vectors:** By default, vectors in ML literature are often assumed to be **column vectors** (elements arranged vertically).
        *   Column vector: $\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}$
        *   A row vector would be written as $\mathbf{x}^T = [x_1, x_2, \ldots, x_n]$ (using the transpose, which we'll cover soon).
*   **Examples in ML:**
    *   A feature vector for a single data sample: $\mathbf{x} = \begin{bmatrix} \text{age} \\ \text{income} \\ \text{years\_experience} \end{bmatrix}$. If age=30, income=50000, years=5, then $\mathbf{x} = \begin{bmatrix} 30 \\ 50000 \\ 5 \end{bmatrix}$.
    *   A word embedding: $\mathbf{e}_{\text{king}} \in \mathbb{R}^{300}$ (a 300-dimensional vector representing the word "king").
    *   The weight vector for a neuron's inputs: $\mathbf{w}$.
    *   The gradient of a loss function with respect to a weight vector: $\nabla L(\mathbf{w})$.

### Matrices
*   **What it is:** A matrix is a 2-dimensional array (a grid) of numbers, arranged in rows and columns. Matrices are used to represent linear transformations, datasets, parameters of neural network layers, and much more.
*   **Notation:**
    *   **Bold uppercase letters:** (e.g., $\mathbf{X}, \mathbf{W}, \mathbf{A}$).
    *   **Elements:** The element in the $i$-th row and $j$-th column of a matrix $\mathbf{W}$ is denoted by $W_{ij}$ or $w_{ij}$ (uppercase or lowercase, italic, not bold, with two subscripts). The first subscript usually denotes the row index, and the second denotes the column index.
    *   **Dimensions:** A matrix with $m$ rows and $n$ columns is described as being in $\mathbb{R}^{m \times n}$.
        *   Example: $\mathbf{W} = \begin{bmatrix} W_{11} & W_{12} \\ W_{21} & W_{22} \\ W_{31} & W_{32} \end{bmatrix}$. This matrix $\mathbf{W} \in \mathbb{R}^{3 \times 2}$ has 3 rows and 2 columns.
*   **Examples in ML:**
    *   A dataset matrix $\mathbf{X}$, where each row is a data sample (feature vector) and each column is a feature: $\mathbf{X} \in \mathbb{R}^{N \times D}$ for $N$ samples and $D$ features.
    *   The weight matrix of a neural network layer: $\mathbf{W}$. If a layer maps an $n$-dimensional input to an $m$-dimensional output, its weight matrix $\mathbf{W}$ will be $m \times n$.
    *   A covariance matrix $\mathbf{\Sigma}$ in statistics.
    *   Query, Key, and Value matrices ($\mathbf{Q}, \mathbf{K}, \mathbf{V}$) in Transformer models.

## 2. Common Operations and Their Notation

### Vector and Matrix Addition/Subtraction
*   **Notation:** $\mathbf{u} + \mathbf{v}$ (vector addition), $\mathbf{A} + \mathbf{B}$ (matrix addition). Subtraction is analogous.
*   **Condition:** The vectors or matrices must have the exact same dimensions. You can't add a $3 \times 2$ matrix to a $2 \times 3$ matrix.
*   **Meaning:** The operation is performed element-wise.
    *   If $\mathbf{C} = \mathbf{A} + \mathbf{B}$, then $C_{ij} = A_{ij} + B_{ij}$ for all $i,j$.
    *   **Example Calculation (Matrix Addition):**
        If $\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ and $\mathbf{B} = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$, then
        $\mathbf{A} + \mathbf{B} = \begin{bmatrix} 1+5 & 2+6 \\ 3+7 & 4+8 \end{bmatrix} = \begin{bmatrix} 6 & 8 \\ 10 & 12 \end{bmatrix}$.
*   **Example in ML:**
    *   Updating weights: $\mathbf{w}_{t+1} = \mathbf{w}_t + \Delta\mathbf{w}_t$.
    *   Adding a bias vector $\mathbf{b}$ to the output of a matrix multiplication: $\mathbf{z} = \mathbf{Wx} + \mathbf{b}$. (Note: $\mathbf{b}$ must have the same dimension as $\mathbf{Wx}$, or broadcasting rules might apply if $\mathbf{b}$ is a scalar or has compatible dimensions, e.g., if $\mathbf{Wx}$ is $m \times 1$ and $\mathbf{b}$ is $m \times 1$).

### Scalar Multiplication (of a vector or matrix)
*   **Notation:** $s\mathbf{v}$ (scalar times vector), $s\mathbf{A}$ (scalar times matrix).
*   **Meaning:** Multiply every element of the vector or matrix by the scalar $s$.
    *   If $\mathbf{B} = s\mathbf{A}$, then $B_{ij} = s \times A_{ij}$ for all $i,j$.
    *   **Example Calculation (Scalar times Matrix):**
        If $s=2$ and $\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$, then
        $s\mathbf{A} = 2 \times \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} = \begin{bmatrix} 2 \times 1 & 2 \times 2 \\ 2 \times 3 & 2 \times 4 \end{bmatrix} = \begin{bmatrix} 2 & 4 \\ 6 & 8 \end{bmatrix}$.
*   **Example in ML:**
    *   The gradient descent update rule: $\mathbf{w} = \mathbf{w} - \eta \nabla L(\mathbf{w})$ (here $\eta$ is a scalar learning rate multiplying the gradient vector $\nabla L(\mathbf{w})$).

### Transpose
*   **Notation:** $\mathbf{A}^T$ (transpose of matrix $\mathbf{A}$), $\mathbf{v}^T$ (transpose of vector $\mathbf{v}$).
*   **Meaning:** Swaps the rows and columns.
    *   If $\mathbf{A}$ is an $m \times n$ matrix, then $\mathbf{A}^T$ is an $n \times m$ matrix.
    *   The element $(A^T)_{ij}$ (element in $i$-th row, $j$-th column of $\mathbf{A}^T$) is $A_{ji}$ (element in $j$-th row, $i$-th column of $\mathbf{A}$).
    *   If $\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix}$ (a $3 \times 1$ column vector), then $\mathbf{v}^T = [v_1, v_2, v_3]$ (a $1 \times 3$ row vector).
    *   **Example Calculation (Matrix Transpose):**
        If $\mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}$ (a $2 \times 3$ matrix), then
        $\mathbf{A}^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}$ (a $3 \times 2$ matrix).
*   **Example in ML:**
    *   Used frequently to make dimensions compatible for multiplication, e.g., $\mathbf{w}^T\mathbf{x}$ for the dot product of two column vectors.
    *   In formulas like $Attention(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$, the transpose of $K$ is used.

### Dot Product (Inner Product of Vectors)
*   **What it is:** A fundamental operation that takes two vectors and returns a single scalar. More specifically, it involves multiplying corresponding elements of the two vectors and then summing up all those products. This single scalar value can represent various concepts, such as the projection of one vector onto another or a measure of their similarity.
*   **Notation:**
    *   $\mathbf{a} \cdot \mathbf{b}$ (less common in ML papers).
    *   $\mathbf{a}^T\mathbf{b}$ (most common, assuming $\mathbf{a}$ and $\mathbf{b}$ are column vectors of the same dimension). If $\mathbf{a} \in \mathbb{R}^n$ (so $\mathbf{a}^T$ is $1 \times n$) and $\mathbf{b} \in \mathbb{R}^n$ (so $\mathbf{b}$ is $n \times 1$), their product $\mathbf{a}^T\mathbf{b}$ is a $1 \times 1$ scalar.
*   **Condition:** The two vectors must have the same number of elements (same dimension).
*   **Meaning:** If $\mathbf{a} = [a_1, \ldots, a_n]^T$ and $\mathbf{b} = [b_1, \ldots, b_n]^T$, then $\mathbf{a}^T\mathbf{b} = \sum_{i=1}^{n} a_i b_i$.
    *   **Example Calculation:** If $\mathbf{a} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$ and $\mathbf{b} = \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix}$, then
        $\mathbf{a}^T\mathbf{b} = (1 \times 4) + (2 \times 5) + (3 \times 6) = 4 + 10 + 18 = 32$.
*   **Example in ML:**
    *   Calculating the input to a neuron (weighted sum of inputs): $z = \mathbf{w}^T\mathbf{x} + b$. Here, $\mathbf{w}^T\mathbf{x}$ is the dot product.
    *   Measuring similarity: Cosine similarity involves a dot product.

### Matrix-Vector Multiplication
*   **Notation:** $\mathbf{y} = \mathbf{Ax}$
*   **Condition for dimensions:** If $\mathbf{A}$ is an $m \times n$ matrix, then $\mathbf{x}$ must be an $n \times 1$ column vector (i.e., have $n$ elements). The number of columns in $\mathbf{A}$ must equal the number of rows in $\mathbf{x}$.
*   **Resulting dimension:** The output $\mathbf{y}$ will be an $m \times 1$ column vector.
*   **Meaning:** Each element $y_i$ of the resulting vector $\mathbf{y}$ is the dot product of the $i$-th row of matrix $\mathbf{A}$ with the vector $\mathbf{x}$.
    *   $y_i = \sum_{j=1}^{n} A_{ij} x_j$.
    *   **Example Calculation:**
        If $\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ and $\mathbf{x} = \begin{bmatrix} 5 \\ 6 \end{bmatrix}$, then
        $\mathbf{y} = \mathbf{Ax} = \begin{bmatrix} (1 \times 5) + (2 \times 6) \\ (3 \times 5) + (4 \times 6) \end{bmatrix} = \begin{bmatrix} 5 + 12 \\ 15 + 24 \end{bmatrix} = \begin{bmatrix} 17 \\ 39 \end{bmatrix}$.
*   **Example in ML:**
    *   A linear transformation in a neural network layer: $\mathbf{z} = \mathbf{Wx} + \mathbf{b}$. $\mathbf{W}$ is the weight matrix, $\mathbf{x}$ is the input vector.

### Matrix-Matrix Multiplication
*   **Notation:** $\mathbf{C} = \mathbf{AB}$
*   **Condition for dimensions:** If $\mathbf{A}$ is an $m \times n$ matrix, then $\mathbf{B}$ must be an $n \times p$ matrix. The number of columns in $\mathbf{A}$ must equal the number of rows in $\mathbf{B}$ (the "inner" dimensions must match).
*   **Resulting dimension:** The output matrix $\mathbf{C}$ will be an $m \times p$ matrix (the "outer" dimensions).
*   **Meaning:** The element $C_{ij}$ (in $i$-th row, $j$-th column of $\mathbf{C}$) is the dot product of the $i$-th row of $\mathbf{A}$ with the $j$-th column of $\mathbf{B}$.
    *   $C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$.
    *   **Example Calculation:**
        If $\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ ($2 \times 2$) and $\mathbf{B} = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$ ($2 \times 2$), then
        $\mathbf{C} = \mathbf{AB} = \begin{bmatrix} (1 \times 5 + 2 \times 7) & (1 \times 6 + 2 \times 8) \\ (3 \times 5 + 4 \times 7) & (3 \times 6 + 4 \times 8) \end{bmatrix} = \begin{bmatrix} 5+14 & 6+16 \\ 15+28 & 18+32 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$.
*   **Important Note:** Matrix multiplication is generally **not commutative**: $\mathbf{AB} \neq \mathbf{BA}$ in most cases.
*   **Example in ML:**
    *   Combining multiple linear transformations: If $\mathbf{z}_1 = \mathbf{W}_1 \mathbf{x}$ and $\mathbf{z}_2 = \mathbf{W}_2 \mathbf{z}_1$, then $\mathbf{z}_2 = \mathbf{W}_2 (\mathbf{W}_1 \mathbf{x}) = (\mathbf{W}_2 \mathbf{W}_1) \mathbf{x}$. Here $\mathbf{W}_{eff} = \mathbf{W}_2 \mathbf{W}_1$ is an effective weight matrix.
    *   The $QK^T$ part of the attention formula.

### Element-wise Product (Hadamard Product)
*   **Notation:** $\mathbf{C} = \mathbf{A} \odot \mathbf{B}$ (for matrices) or $\mathbf{c} = \mathbf{a} \odot \mathbf{b}$ (for vectors).
*   **Condition:** The matrices (or vectors) $\mathbf{A}$ and $\mathbf{B}$ must have the exact same dimensions.
*   **Meaning:** Each element of the resulting matrix (or vector) $\mathbf{C}$ is the product of the corresponding elements of $\mathbf{A}$ and $\mathbf{B}$.
    *   $C_{ij} = A_{ij} \times B_{ij}$.
    *   **Example Calculation (Hadamard Product of Matrices):**
        If $\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ and $\mathbf{B} = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$, then
        $\mathbf{A} \odot \mathbf{B} = \begin{bmatrix} 1 \times 5 & 2 \times 6 \\ 3 \times 7 & 4 \times 8 \end{bmatrix} = \begin{bmatrix} 5 & 12 \\ 21 & 32 \end{bmatrix}$.
*   **Important:** This is **different** from matrix multiplication $\mathbf{AB}$.
*   **Example in ML:**
    *   Gating mechanisms in LSTMs or GRUs often use Hadamard products.
    *   If $f$ is a function like $\text{sigmoid}$ or $\text{ReLU}$, applying it to a matrix $\mathbf{Z}$ as $f(\mathbf{Z})$ often implies an element-wise application of $f$ to each element of $\mathbf{Z}$. While not strictly a Hadamard product *between two matrices*, it's an element-wise operation.

## 3. Special Vectors and Matrices

*   **Zero Vector/Matrix ($\mathbf{0}$):** A vector or matrix where all elements are zero. Its dimensions are usually clear from context.
    *   Example: $\mathbf{w} = \mathbf{0}$ (initializing weights to zero).
*   **Ones Vector/Matrix ($\mathbf{1}$):** A vector or matrix where all elements are one.
*   **Identity Matrix ($\mathbf{I}$ or $\mathbf{I}_n$):** A square matrix ($n \times n$) with ones on the main diagonal (from top-left to bottom-right) and zeros everywhere else.
    *   It's the matrix equivalent of the number 1: For any matrix $\mathbf{A}$ (where dimensions are compatible), $\mathbf{AI} = \mathbf{A}$ and $\mathbf{IA} = \mathbf{A}$.
    *   $\mathbf{I}_3 = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$.
*   **Diagonal Matrix:** A square matrix where all elements off the main diagonal are zero. The diagonal elements can be any value.
    *   Notation: $\text{diag}(d_1, d_2, \ldots, d_n)$ or $\text{diag}(\mathbf{d})$ creates a diagonal matrix with the elements of vector $\mathbf{d}$ on its diagonal.

## 4. Vector Norms: Measuring Length/Magnitude

A norm is a function that assigns a strictly positive "length" or "size" to a vector (except for the zero vector, which has zero length).

*   **Notation:** $\|\mathbf{x}\|$ or $\|\mathbf{x}\|_p$ for the $L_p$ norm.
*   **Common Norms:**
    *   **$L_2$ Norm (Euclidean Norm):** Most common. It's the standard geometric length of a vector.
        *   $\|\mathbf{x}\|_2 = \sqrt{x_1^2 + x_2^2 + \ldots + x_n^2} = \sqrt{\sum_{i=1}^{n} x_i^2}$.
        *   Often written simply as $\|\mathbf{x}\|$.
        *   Note that $\|\mathbf{x}\|_2^2 = \mathbf{x}^T\mathbf{x}$ (the squared $L_2$ norm is the dot product of the vector with itself).
    *   **$L_1$ Norm (Manhattan Norm):** Sum of the absolute values of the elements.
        *   $\|\mathbf{x}\|_1 = \sum_{i=1}^{n} |x_i|$.
*   **Examples in ML:**
    *   **Regularization:** $L_1$ and $L_2$ norms are used in regularization terms to penalize large weights (e.g., adding $\lambda \|\mathbf{w}\|_2^2$ or $\lambda \|\mathbf{w}\|_1$ to the loss function).
    *   **Distance calculation:** The Euclidean distance between two vectors $\mathbf{a}$ and $\mathbf{b}$ is $\|\mathbf{a}-\mathbf{b}\|_2$.
    *   **Normalization:** Dividing a vector by its norm to get a unit vector (length 1): $\hat{\mathbf{x}} = \frac{\mathbf{x}}{\|\mathbf{x}\|_2}$.

## 5. Reading Formulas with Linear Algebra: An Example

Let's look at a typical formula for a single neuron's output or a simple linear layer (without activation function for now):
$\mathbf{y} = \mathbf{Wx} + \mathbf{b}$

Let's break this down:
*   $\mathbf{x} \in \mathbb{R}^D$: Input feature vector (e.g., $D$ features for a data sample).
*   $\mathbf{W} \in \mathbb{R}^{K \times D}$: Weight matrix. It maps from $D$-dimensional input to $K$-dimensional output.
*   $\mathbf{b} \in \mathbb{R}^K$: Bias vector.
*   $\mathbf{y} \in \mathbb{R}^K$: Output vector.

**Dimensional Analysis:**
1.  $\mathbf{Wx}$: Matrix $\mathbf{W}$ ($K \times D$) multiplies vector $\mathbf{x}$ ($D \times 1$). The inner dimensions ($D$ and $D$) match. The result is a $K \times 1$ vector.
2.  $\mathbf{Wx} + \mathbf{b}$: The result of $\mathbf{Wx}$ ($K \times 1$) is added to vector $\mathbf{b}$ ($K \times 1$). The dimensions match for element-wise addition.
3.  The final output $\mathbf{y}$ is a $K \times 1$ vector.

Understanding the dimensions is crucial for correctly interpreting and implementing these formulas. If the dimensions don't line up for an operation, the formula is usually incorrect or requires a special interpretation (like broadcasting).
