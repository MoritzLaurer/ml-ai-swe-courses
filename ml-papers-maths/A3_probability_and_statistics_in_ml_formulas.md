# A3: Probability and Statistics in ML Formulas

Probability and statistics are the bedrock upon which much of machine learning is built. They provide the tools to quantify uncertainty, make inferences from data, and understand the behavior of our models. This module introduces key notation and concepts from probability and statistics that you'll frequently encounter in ML papers.

## 1. Basic Probability Notation

Probability theory deals with the likelihood of events occurring.

*   **Probability of an Event $P(A)$**:
    *   **Notation:** $P(A)$
    *   **Meaning:** The probability that event $A$ occurs. Probability values range from 0 (impossible event) to 1 (certain event).
    *   **Example:** If you flip a fair coin, the event $A$ could be "getting heads." Then $P(A) = 0.5$.
    *   **In ML:** We often talk about the probability of a data point belonging to a certain class, e.g., $P(\text{class} = \text{spam} | \text{email features})$.

*   **Conditional Probability $P(A|B)$**:
    *   **Notation:** $P(A|B)$
    *   **Meaning:** The probability that event $A$ occurs *given that* event $B$ has already occurred. Read as "the probability of A given B."
    *   **Formal Definition/Calculation:** The value of $P(A|B)$ is formally defined and calculated as $P(A|B) = \frac{P(A \cap B)}{P(B)}$, where $P(A \cap B)$ is the probability that both A and B occur, and $P(B)$ (the probability of the given event) must be greater than 0.
    *   **Real-world Example:** What is the probability it will rain today ($A$) *given that* there are dark clouds in the sky ($B$)? This is $P(\text{Rain} | \text{Dark Clouds})$. This probability is likely higher than $P(\text{Rain})$ without any prior information.
    *   **In ML:** This is fundamental.
        *   Language models predict the next word given previous words: $P(\text{next word} | \text{previous words})$.
        *   Classification models output the probability of a class given input features: $P(Y=c | \mathbf{X}=\mathbf{x})$.
            *   Here, $\mathbf{X}$ represents the random variable for all possible input features, and $\mathbf{x}$ is a specific observed set of feature values.
            *   $Y$ represents the random variable for the true class label, and $c$ is a specific class (e.g., 'spam', 'not spam'). So, this reads: "the probability that the true class label $Y$ is $c$, given that the input features $\mathbf{X}$ are observed as $\mathbf{x}$."

## 2. Random Variables (RVs)

A random variable is a variable whose value is a numerical outcome of a random phenomenon.

*   **Notation:**
    *   **Random Variable itself:** Usually an uppercase letter (e.g., $X, Y, Z$). This represents the concept or the process of generating a value.
    *   **Realization/Specific Value:** Usually a lowercase letter (e.g., $x, y, z$). This represents a specific outcome or observed value of the random variable.
*   **Meaning:** $X$ could represent "the outcome of a dice roll." Then $x$ could be any specific value from $\{1, 2, 3, 4, 5, 6\}$. We can ask $P(X=x)$, e.g., $P(X=3) = 1/6$ for a fair die.
*   **Types (briefly):**
    *   **Discrete RV:** Takes on a finite or countably infinite number of values (e.g., outcome of a coin flip {0, 1}, number of heads in 10 flips, word identity from a vocabulary).
    *   **Continuous RV:** Takes on any value within a continuous range (e.g., height of a person, temperature, a pixel intensity value between 0 and 1).
*   **In ML:**
    *   Input features $\mathbf{X}$ can be considered random variables. A specific data instance is a realization $\mathbf{x}$.
    *   Model predictions $\hat{Y}$ are often treated as random variables. (Note: $\hat{Y}$ is commonly read "Y-hat" and the "hat" symbol typically denotes an estimated or predicted value).
    *   The parameters $\theta$ of a model might be treated as random variables in Bayesian machine learning.

## 3. Expected Value ($\mathbb{E}$)

The expected value is, intuitively, the long-run average value of a random variable if you were to repeat the random process many times.

*   **Notation:**
    *   $\mathbb{E}[X]$: The expected value of the random variable $X$.
    *   $\mathbb{E}_{x \sim P}[f(x)]$ or $\mathbb{E}_{X \sim P(X)}[f(X)]$: The expected value of a function $f(x)$ where $x$ is drawn from a probability distribution $P$. This notation emphasizes that the expectation is taken with respect to the distribution $P$. Sometimes, if the distribution $P$ is clear from context, it might be written as $\mathbb{E}_x[f(x)]$ or just $\mathbb{E}[f(X)]$.
*   **Meaning:**
    *   **For a discrete RV $X$** that can take values $x_1, x_2, \ldots, x_k$ with probabilities $P(X=x_1), P(X=x_2), \ldots, P(X=x_k)$, the expected value is:
        $\mathbb{E}[X] = \sum_{i=1}^{k} x_i P(X=x_i)$.
        (This formula defines a weighted average. Each possible value $x_i$ is "weighted" by its corresponding probability $P(X=x_i)$. Values with higher probabilities contribute more to the sum. The probabilities $P(X=x_i)$ act as the weights, and they sum to 1. If, as a special case, all $k$ outcomes were equally likely, then each $P(X=x_i)$ would be $1/k$, and the formula would simplify to the standard arithmetic mean: $\frac{1}{k}\sum_{i=1}^{k} x_i$.)
    *   **For a continuous RV $X$** with probability density function $p(x)$ (think of $p(x)dx$ as the probability $X$ falls in a tiny interval around $x$), the expected value is:
        $\mathbb{E}[X] = \int_{-\infty}^{\infty} x p(x) dx$. (This is an integral, which is like a continuous sum).
*   **Real-world Example:** If you roll a fair six-sided die ($X$), the possible outcomes are $\{1, 2, 3, 4, 5, 6\}$, each with probability $1/6$.
    $\mathbb{E}[X] = (1 \times \frac{1}{6}) + (2 \times \frac{1}{6}) + (3 \times \frac{1}{6}) + (4 \times \frac{1}{6}) + (5 \times \frac{1}{6}) + (6 \times \frac{1}{6}) = \frac{1+2+3+4+5+6}{6} = \frac{21}{6} = 3.5$.
    Notice that the expected value (3.5) is not necessarily one of the possible outcomes.
*   **In ML:**
    *   **Loss Functions:** The goal of training is often to minimize an expected loss, e.g., $\mathbb{E}_{(\mathbf{x}, y) \sim D}[\mathcal{L}(f(\mathbf{x}; \theta), y)]$.
        *   Here, $(\mathbf{x}, y)$ is a data sample consisting of an input feature vector $\mathbf{x}$ and its true label $y$.
        *   The notation $\sim D$ signifies that this sample is drawn from the true underlying data distribution $D$.
        *   $\mathcal{L}$ is the loss function, measuring the error between the prediction and the true label.
        *   $f(\mathbf{x}; \theta)$ is the model's prediction for input $\mathbf{x}$, given its parameters $\theta$.
        *   $\theta$ represents the parameters of the model that are learned during training.
        We approximate this true expected loss by averaging the loss over our finite training dataset.
    *   **Reinforcement Learning (RL):** The value of a state $V^\pi(s)$ is the expected sum of future rewards starting from state $s$: $V^\pi(s) = \mathbb{E}_\pi [ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t=s ]$.
        *   $V^\pi(s)$ is the value (expected long-term reward) of being in state $s$ if the agent follows policy $\pi$.
        *   $\mathbb{E}_\pi$ denotes the expectation when the agent makes decisions according to policy $\pi$.
        *   $\sum_{k=0}^{\infty}$ is the sum of rewards over all future time steps (from the current time $t$ to infinity).
        *   $\gamma^k$ is the discount factor $\gamma$ (a value between 0 and 1) raised to the power of $k$. This makes rewards obtained further in the future less valuable than immediate rewards.
        *   $R_{t+k+1}$ is the reward received at the future time step $t+k+1$.
        *   $S_t=s$ indicates that the expectation is conditioned on the agent currently being in state $s$ at time step $t$.

## 4. Variance ($\text{Var}$)

Variance measures how spread out the values of a random variable are from its expected value (its mean). A low variance means values tend to be close to the mean; high variance means values are spread out.

*   **Notation:**
    *   $\text{Var}(X)$
    *   $\sigma^2$ (sigma squared, where $\sigma$ is the standard deviation)
*   **Meaning:** The variance is defined as the expected value of the squared difference between the random variable and its mean $\mu = \mathbb{E}[X]$.
    *   $\text{Var}(X) = \mathbb{E}[(X - \mu)^2] = \mathbb{E}[(X - \mathbb{E}[X])^2]$.
    *   A common computational formula is $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$.
*   **Standard Deviation ($\sigma$):** This is simply the square root of the variance: $\sigma = \sqrt{\text{Var}(X)}$. The standard deviation is often preferred because it has the same units as the random variable $X$.
*   **Real-world Example:**
    *   Consider two investment options. Both have an expected return of $10\%$.
        *   Investment A has returns that usually fall between $8\%$ and $12\%$ (low variance).
        *   Investment B has returns that can swing wildly from $-20\%$ to $+40\%$ (high variance).
    *   While the average return is the same, Investment B is much riskier due to its higher variance.
*   **In ML:**
    *   **Model Evaluation:** When training a model multiple times (e.g., with different random initializations), looking at the variance of its performance metrics (like accuracy) can tell you how stable the model's training process is.
    *   **Bias-Variance Tradeoff:** A key concept in ML. Models with high variance are sensitive to the specific training data and may overfit (perform well on training data but poorly on new data).
    *   **Uncertainty Quantification:** Variance can be a component in estimating the uncertainty of model predictions.

## 5. Common Probability Distributions and Their Notation

A probability distribution describes the probabilities of occurrence of different possible outcomes for a random variable.

*   **The "Tilde" Notation $X \sim P$**:
    *   This notation $X \sim P$ is read as "the random variable $X$ is distributed according to (or follows) the probability distribution $P$."
    *   Often, $P$ will be a named distribution with its parameters.

Here are a few common ones:

*   **Normal (Gaussian) Distribution:** $\mathcal{N}(\mu, \sigma^2)$
    *   **Notation:** $X \sim \mathcal{N}(\mu, \sigma^2)$
    *   **Meaning:** $X$ follows a Normal distribution with mean $\mu$ and variance $\sigma^2$. This is the classic "bell curve."
    *   **Parameters:**
        *   $\mu$ (mu): The mean or center of the distribution.
        *   $\sigma^2$ (sigma squared): The variance (spread). $\sigma$ is the standard deviation.
    *   **In ML:** Very common. Used to model errors, initialize weights in neural networks, represent beliefs about parameters in Bayesian models, etc. The Central Limit Theorem states that the sum of many independent random variables tends towards a Normal distribution.

*   **Bernoulli Distribution:** $\text{Bernoulli}(p)$
    *   **Notation:** $X \sim \text{Bernoulli}(p)$
    *   **Meaning:** Represents an experiment with two outcomes (e.g., success/failure, 0/1, head/tail).
    *   **Parameter:** $p$ is the probability of one outcome (e.g., success, or $X=1$). The probability of the other outcome ($X=0$) is $1-p$.
    *   **In ML:** Used for binary classification outputs (e.g., probability of an email being spam). A single data point in a binary classification problem can be thought of as a Bernoulli trial.

*   **Categorical Distribution:** $\text{Cat}(\mathbf{p})$ or $\text{Multinoulli}(\mathbf{p})$
    *   **Notation:** $X \sim \text{Cat}(\mathbf{p})$ where $\mathbf{p} = (p_1, p_2, \ldots, p_K)$.
    *   **Meaning:** Generalization of the Bernoulli distribution to $K$ possible outcomes (categories), where each outcome $k$ has a probability $p_k$. The sum of all $p_k$ must be 1. $X$ takes a value from $\{1, ..., K\}$.
    *   **Parameter:** $\mathbf{p}$ is a vector of probabilities for each of the $K$ categories.
    *   **In ML:** Used for multi-class classification where an item belongs to one of $K$ distinct classes (e.g., classifying an image as 'cat', 'dog', or 'bird'). The output of a softmax function in a classifier for a single instance follows a categorical distribution.

*   **Uniform Distribution:** $\mathcal{U}(a, b)$
    *   **Notation:** $X \sim \mathcal{U}(a, b)$
    *   **Meaning:** All values in a continuous range between $a$ and $b$ are equally likely. For a discrete uniform distribution, all $n$ values are equally likely with probability $1/n$.
    *   **Parameters:** $a$ (lower bound) and $b$ (upper bound).
    *   **In ML:** Used for random initialization of parameters when there's no prior reason to prefer certain values, or for sampling from a range where all options should be equally probable.

## 6. Argmax and Argmin Notation

These are not probabilities themselves, but they are often used in conjunction with probability distributions or other functions in ML.

*   **Argmax ($\text{argmax}_x f(x)$)**:
    *   **Notation:** $\text{argmax}_x f(x)$
    *   **Meaning:** "The argument (or value) of $x$ that maximizes the function $f(x)$." It doesn't return the maximum value of $f(x)$, but rather the $x$ that *produces* that maximum value.
    *   **Example:** If $f(x) = -(x-2)^2 + 10$, the function $f(x)$ has a maximum value of 10 when $x=2$. So, $\text{argmax}_x f(x) = 2$.
    *   **In ML:**
        *   In classification, if a model outputs probabilities $P(y=c|\mathbf{x})$ for each class $c$, the predicted class is often $\hat{y} = \text{argmax}_c P(y=c|\mathbf{x})$. (Choose the class with the highest probability).
        *   Finding the action that maximizes expected reward in RL.

*   **Argmin ($\text{argmin}_x f(x)$)**:
    *   **Notation:** $\text{argmin}_x f(x)$
    *   **Meaning:** "The argument (or value) of $x$ that minimizes the function $f(x)$."
    *   **Example:** If $f(x) = (x-3)^2 + 5$, the function $f(x)$ has a minimum value of 5 when $x=3$. So, $\text{argmin}_x f(x) = 3$.
    *   **In ML:**
        *   Most model training involves finding parameters $\theta$ that minimize a loss function $\mathcal{L}(\theta)$: $\hat{\theta} = \text{argmin}_\theta \mathcal{L}(\theta)$.

## 7. Notation for Sampling

Sampling refers to the process of drawing one or more observations (realizations) from a probability distribution or a dataset.

*   **Notation:**
    *   $x \sim D$: Read as "$x$ is sampled from dataset $D$."
    *   $x \sim P(X)$ or $x \sim P$: Read as "$x$ is sampled from the probability distribution $P(X)$ (or simply $P$ if $X$ is implied)."
    *   Sometimes, a set of samples is denoted: $\{x^{(1)}, x^{(2)}, \ldots, x^{(N)}\} \sim P(X)$, meaning $N$ independent and identically distributed (i.i.d.) samples drawn from $P(X)$.
*   **Real-world Example:**
    *   Drawing a card from a deck: The card drawn is a sample from the uniform distribution over the 52 cards.
    *   Surveying 100 people about their voting preference: Each person's response is a sample.
*   **In ML:**
    *   **Training Data:** We train models on a sample (dataset) $D$ drawn from some true underlying (but unknown) data distribution $P_{\text{data}}$. E.g., $(\mathbf{x}^{(i)}, y^{(i)}) \sim D$.
    *   **Stochastic Gradient Descent (SGD):** Minibatches of data are sampled from the training set at each iteration.
    *   **Monte Carlo Methods:** Used extensively in RL and Bayesian methods, involve drawing samples to approximate expectations or distributions. E.g., estimating $\mathbb{E}[f(X)]$ by drawing samples $x_1, \ldots, x_N \sim P(X)$ and calculating $\frac{1}{N}\sum f(x_i)$.
    *   **Generative Models:** A trained generative model (like a GAN or VAE) learns a distribution $P_{\text{model}}(\mathbf{x})$, and then we can sample $\mathbf{x}_{\text{new}} \sim P_{\text{model}}(\mathbf{x})$ to create new data instances (e.g., new images).

Understanding these probabilistic and statistical notations will greatly aid in deciphering the formulas that describe how ML models learn from data and make predictions under uncertainty.