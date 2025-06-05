# A4: Calculus in ML Formulas

Calculus is the mathematical foundation that makes machine learning optimization possible. While we briefly introduced derivatives and integrals in Module A1, this module focuses specifically on how calculus appears in ML contexts. Most importantly, we'll explore how derivatives enable models to "learn" by finding the best parameters through optimization.

## 1. Review: Derivatives and Their ML Significance

In Module A1, we learned that a derivative $\frac{df}{dx}$ measures how much a function $f(x)$ changes when we make a tiny change to $x$. In machine learning, this concept is fundamental because:

*   **Models learn by adjusting parameters:** ML models have parameters (like weights $w$ and biases $b$) that determine their predictions.
*   **We measure performance with loss functions:** A loss function $L$ measures how "wrong" our model's predictions are.
*   **Derivatives tell us how to improve:** The derivative $\frac{dL}{dw}$ tells us how the loss changes when we adjust a weight $w$. If this derivative is positive, increasing $w$ increases the loss (makes things worse). If it's negative, increasing $w$ decreases the loss (makes things better).

**Real-world Analogy:** Imagine you're hiking and want to reach the bottom of a valley (minimize elevation). The derivative is like the slope of the ground under your feet. If the slope points upward to your right, you should walk left to go downhill. If it points downward to your right, you should walk right to go downhill.

## 2. Partial Derivatives: When Functions Have Multiple Variables

Most ML functions depend on many variables simultaneously. A neural network might have millions of parameters! When a function depends on multiple variables, we use **partial derivatives**.

*   **Notation:** $\frac{\partial f}{\partial x}$ (read as "partial f with respect to x")
*   **Meaning:** How much $f$ changes when we change $x$, *while keeping all other variables constant*.
*   **The $\partial$ Symbol:** This is the "partial" symbol (as opposed to $d$ for ordinary derivatives). It reminds us that we're only varying one variable at a time.

**Example:** Consider a simple loss function for linear regression:
$$L(w, b) = \frac{1}{2}(y - (wx + b))^2$$

Here, $L$ depends on both the weight $w$ and the bias $b$. We can compute:
*   $\frac{\partial L}{\partial w}$: How the loss changes when we adjust the weight (keeping bias fixed)
*   $\frac{\partial L}{\partial b}$: How the loss changes when we adjust the bias (keeping weight fixed)

**Real-world Analogy:** Imagine you're adjusting the temperature and humidity in a greenhouse to optimize plant growth. The partial derivative with respect to temperature tells you how growth changes when you adjust *only* the temperature (keeping humidity fixed). The partial derivative with respect to humidity tells you how growth changes when you adjust *only* humidity (keeping temperature fixed).

### Common Notation Variations

You'll see several ways to write partial derivatives in ML papers:

*   $\frac{\partial L}{\partial w_i}$: Partial derivative of $L$ with respect to the $i$-th weight
*   $\frac{\partial}{\partial \theta_j} J(\theta)$: Partial derivative of function $J$ with respect to the $j$-th parameter in vector $\theta$
*   $L_{w_i}$ or $L_w$: Shorthand notation (less common in ML papers)

## 3. Gradients: The Vector of All Partial Derivatives

When a function depends on multiple variables, the **gradient** collects all the partial derivatives into a single vector.

*   **Notation:** $\nabla f$ or $\nabla_x f$ (when we want to emphasize the variable)
*   **The $\nabla$ Symbol:** This is called "nabla" or "del." It's the gradient operator.
*   **Meaning:** If $f$ depends on variables $x_1, x_2, \ldots, x_n$, then:
    $$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

**Key Insight:** The gradient points in the direction of *steepest increase* of the function. To minimize the function (reduce loss), we want to move in the *opposite* direction of the gradient.

### Gradient Notation in ML

*   **Loss with respect to parameters:** $\nabla_\theta L(\theta)$ or $\nabla L(\theta)$
    *   This is a vector containing $\frac{\partial L}{\partial \theta_1}, \frac{\partial L}{\partial \theta_2}, \ldots$ for all parameters
*   **Cost function gradients:** $\nabla J(\mathbf{w})$ where $J$ is the cost function and $\mathbf{w}$ is the weight vector
*   **Gradient with respect to inputs:** $\nabla_\mathbf{x} f(\mathbf{x})$ (used in adversarial examples, saliency maps)

**Example:** For the linear regression loss $L(w, b) = \frac{1}{2}(y - (wx + b))^2$:
$$\nabla L = \begin{bmatrix} \frac{\partial L}{\partial w} \\ \frac{\partial L}{\partial b} \end{bmatrix} = \begin{bmatrix} -(y - (wx + b)) \cdot x \\ -(y - (wx + b)) \end{bmatrix}$$

## 4. The Chain Rule: Computing Derivatives of Composite Functions

The chain rule is arguably the most important calculus concept in ML because it enables **backpropagation** - the algorithm that trains neural networks.

*   **Basic Form:** If $y = f(u)$ and $u = g(x)$, then $\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$
*   **Intuition:** To see how $y$ changes with respect to $x$, we track how $y$ changes with respect to the intermediate variable $u$, and how $u$ changes with respect to $x$.

### Chain Rule in Neural Networks

Neural networks are compositions of many functions. For a simple two-layer network:
$$\mathbf{z}_1 = \mathbf{W}_1\mathbf{x} + \mathbf{b}_1$$
$$\mathbf{a}_1 = \sigma(\mathbf{z}_1)$$
$$\mathbf{z}_2 = \mathbf{W}_2\mathbf{a}_1 + \mathbf{b}_2$$
$$L = \text{loss}(\mathbf{z}_2, \mathbf{y})$$

To compute $\frac{\partial L}{\partial \mathbf{W}_1}$ (how the loss changes with respect to the first layer weights), we use the chain rule:
$$\frac{\partial L}{\partial \mathbf{W}_1} = \frac{\partial L}{\partial \mathbf{z}_2} \cdot \frac{\partial \mathbf{z}_2}{\partial \mathbf{a}_1} \cdot \frac{\partial \mathbf{a}_1}{\partial \mathbf{z}_1} \cdot \frac{\partial \mathbf{z}_1}{\partial \mathbf{W}_1}$$

**Real-world Analogy:** Imagine a factory assembly line where raw materials go through multiple processing steps before becoming the final product. If you want to know how changing the first step affects the final quality, you need to trace the effect through each intermediate step. The chain rule is like tracking this cause-and-effect chain.

### Multivariate Chain Rule

For functions of multiple variables (which is the norm in ML), the chain rule becomes:
$$\frac{\partial f}{\partial x_i} = \sum_j \frac{\partial f}{\partial u_j} \cdot \frac{\partial u_j}{\partial x_i}$$

This summation accounts for all the pathways through which changing $x_i$ can affect $f$.

## 5. Optimization and Gradient Descent

The primary reason we compute gradients in ML is to optimize (minimize) loss functions. **Gradient descent** is the fundamental optimization algorithm.

*   **Basic Update Rule:** $\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$
*   **Components:**
    *   $\theta_t$: Current parameter values at iteration $t$
    *   $\theta_{t+1}$: Updated parameter values at iteration $t+1$
    *   $\eta$: Learning rate (how big steps to take)
    *   $\nabla L(\theta_t)$: Gradient of the loss with respect to parameters
    *   The minus sign: We move *opposite* to the gradient direction (toward lower loss)

### Why the Minus Sign?

*   The gradient $\nabla L(\theta)$ points in the direction where the loss *increases* most rapidly
*   We want to *decrease* the loss, so we move in the opposite direction
*   Hence: $\theta_{\text{new}} = \theta_{\text{old}} - \eta \nabla L(\theta_{\text{old}})$

**Real-world Analogy:** If you're hiking and want to reach the bottom of a valley:
*   The gradient tells you which direction is steepest uphill
*   To go downhill, you walk in the opposite direction
*   The learning rate $\eta$ determines how big steps you take

### Variations You'll See in Papers

*   **Stochastic Gradient Descent (SGD):** $\theta_{t+1} = \theta_t - \eta \nabla L_{\text{batch}}(\theta_t)$
    *   Uses gradient computed on a mini-batch of data rather than the full dataset
*   **Momentum:** $\theta_{t+1} = \theta_t - \eta \mathbf{m}_t$ where $\mathbf{m}_t$ incorporates previous gradients
*   **Adam and variants:** More complex update rules that adapt the learning rate

## 6. Common Derivative Formulas in ML

Here are some derivatives you'll frequently encounter in ML papers:

### Activation Functions
*   **Sigmoid:** $\sigma(x) = \frac{1}{1 + e^{-x}}$, so $\frac{d\sigma}{dx} = \sigma(x)(1 - \sigma(x))$
*   **ReLU:** $\text{ReLU}(x) = \max(0, x)$, so $\frac{d\text{ReLU}}{dx} = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$
*   **Tanh:** $\tanh(x)$, so $\frac{d\tanh}{dx} = 1 - \tanh^2(x)$

### Loss Functions
*   **Squared Error:** $L = \frac{1}{2}(y - \hat{y})^2$, so $\frac{\partial L}{\partial \hat{y}} = -(y - \hat{y})$
*   **Cross-Entropy:** $L = -y \log(\hat{y}) - (1-y) \log(1-\hat{y})$, so $\frac{\partial L}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}$

### Matrix Operations
*   **Linear transformation:** If $\mathbf{z} = \mathbf{Wx} + \mathbf{b}$, then:
    *   $\frac{\partial \mathbf{z}}{\partial \mathbf{W}} = \mathbf{x}^T$ (often written as an outer product)
    *   $\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = \mathbf{W}^T$
    *   $\frac{\partial \mathbf{z}}{\partial \mathbf{b}} = \mathbf{I}$ (identity matrix)

## 7. Integrals in ML Contexts

While derivatives are more common in practical ML, integrals appear in theoretical contexts:

### Marginalization
*   **Continuous case:** $p(x) = \int p(x,y) dy$
*   **Meaning:** To get the probability distribution of $x$ alone, we "integrate out" (marginalize over) the variable $y$
*   **Discrete analogy:** $p(x) = \sum_y p(x,y)$

### Expected Values
*   **Definition:** $\mathbb{E}[f(X)] = \int f(x) p(x) dx$ for continuous $X$
*   **In ML:** Many objectives are expected values, like $\mathbb{E}_{(\mathbf{x},y) \sim D}[L(f(\mathbf{x}), y)]$

### Bayesian ML
*   **Posterior computation:** $p(\theta | D) = \frac{p(D | \theta) p(\theta)}{\int p(D | \theta') p(\theta') d\theta'}$
*   **The denominator is an integral** that normalizes the posterior distribution

### Variational Inference
*   **Evidence Lower Bound (ELBO):** Often involves integrals like $\int q(\mathbf{z}) \log p(\mathbf{x}, \mathbf{z}) d\mathbf{z}$
*   **These are approximated** using sampling methods or reparameterization tricks

## 8. Reading Calculus Notation in Papers

### Subscripts and Superscripts in Derivatives

*   **Subscripts often indicate which variable:**
    *   $L_{\theta}$ or $L_\theta$ might mean $\frac{\partial L}{\partial \theta}$
    *   $f_x(x,y)$ means $\frac{\partial f}{\partial x}$ evaluated at point $(x,y)$

*   **Multiple subscripts for higher-order derivatives:**
    *   $f_{xy}$ means $\frac{\partial^2 f}{\partial x \partial y}$ (mixed partial derivative)
    *   $f_{xx}$ means $\frac{\partial^2 f}{\partial x^2}$ (second derivative with respect to $x$)

*   **Time derivatives in optimization:**
    *   $\dot{\theta}$ or $\frac{d\theta}{dt}$ for continuous-time gradient flow
    *   $\theta^{(t)}$ for discrete time steps in iterative algorithms

### Gradient Notation Variations

*   **With respect to specific variables:**
    *   $\nabla_\theta J$ emphasizes that we're taking the gradient with respect to $\theta$
    *   $\nabla_{\mathbf{x}} f(\mathbf{x}, \mathbf{y})$ takes the gradient only with respect to $\mathbf{x}$, treating $\mathbf{y}$ as constant

*   **Evaluated at specific points:**
    *   $\nabla f(\mathbf{x}_0)$ means the gradient evaluated at the specific point $\mathbf{x}_0$
    *   $\left.\frac{\partial f}{\partial x}\right|_{x=a}$ means the partial derivative evaluated at $x=a$

### Operator Notation

*   **Laplacian:** $\nabla^2 f$ or $\Delta f = \sum_i \frac{\partial^2 f}{\partial x_i^2}$ (sum of second partial derivatives)
*   **Divergence:** $\nabla \cdot \mathbf{F}$ for vector field $\mathbf{F}$ (appears in some physics-inspired ML methods)
*   **Directional derivatives:** $\nabla_{\mathbf{v}} f$ means the derivative of $f$ in the direction of vector $\mathbf{v}$

## 9. Putting It All Together: Reading an Optimization Formula

Let's decode a typical optimization formula you might see in an ML paper:
$$\theta^{(t+1)} = \theta^{(t)} - \eta_t \nabla_\theta \mathbb{E}_{(\mathbf{x},y) \sim D}[L(f(\mathbf{x}; \theta), y)]$$

**Breaking it down:**
*   $\theta^{(t+1)}$: Parameters at the next iteration
*   $\theta^{(t)}$: Parameters at the current iteration  
*   $\eta_t$: Learning rate at iteration $t$ (might change over time)
*   $\nabla_\theta$: Gradient with respect to parameters $\theta$
*   $\mathbb{E}_{(\mathbf{x},y) \sim D}[\cdot]$: Expected value over data distribution $D$
*   $L(f(\mathbf{x}; \theta), y)$: Loss function comparing model prediction $f(\mathbf{x}; \theta)$ to true label $y$

**In plain English:** "Update the parameters by moving in the direction opposite to the gradient of the expected loss over the data distribution."

**In practice:** We approximate the expectation using a mini-batch of data, so this becomes:
$$\theta^{(t+1)} = \theta^{(t)} - \eta_t \frac{1}{|B|} \sum_{(\mathbf{x},y) \in B} \nabla_\theta L(f(\mathbf{x}; \theta^{(t)}), y)$$

Understanding calculus notation in ML is fundamentally about recognizing that most formulas describe how to compute and use derivatives to improve model parameters. Once you grasp this central concept, the specific notation becomes much more approachable.