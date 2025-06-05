# -*- coding: utf-8 -*-
# 14_loss_functions_for_llms.py

# Module 14: Understanding and Combining Loss Functions in LLMs
#
# This script provides an overview of common loss functions used in
# training and fine-tuning Large Language Models (LLMs). It also
# demonstrates how to combine multiple loss functions in PyTorch.
#
# We will cover:
# 1.  The Role of Loss Functions.
# 2.  Common Loss Functions for LLMs:
#     - Cross-Entropy Loss (nn.CrossEntropyLoss)
#     - Negative Log-Likelihood Loss (nn.NLLLoss)
#     - Mean Squared Error Loss (nn.MSELoss)
#     - KL Divergence Loss (nn.KLDivLoss)
#     - Conceptual mentions of other losses.
# 3.  Combining Loss Functions.
# 4.  Concrete PyTorch examples for each.

import torch
import torch.nn as nn
import torch.nn.functional as F # For functional equivalents like softmax

print("--- Module 14: Understanding and Combining Loss Functions in LLMs ---\n")
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


# %% 1. The Role of Loss Functions
print("\n--- 1. The Role of Loss Functions ---\n")

print("A loss function, also known as a cost function or objective function, quantifies how well")
print("a machine learning model performs on a given task. It measures the discrepancy (the 'loss')")
print("between the model's predictions and the actual target values (ground truth).")
print("\nThe primary goal during the training process is to adjust the model's parameters (weights and biases)")
print("in such a way that this loss is minimized. This is typically achieved using optimization algorithms")
print("like Stochastic Gradient Descent (SGD) or Adam, which use the gradients of the loss function")
print("with respect to the model parameters.")
print("\nThe choice of an appropriate loss function is critical as it directly influences what the model learns")
print("and how it behaves. The loss function must be tailored to the specific nature of the task")
print("(e.g., classification, regression, generation) and the desired properties of the model's output.")


# %% 2. Common Loss Functions for LLMs
print("\n--- 2. Common Loss Functions for LLMs ---\n")

# --- 2.1. Cross-Entropy Loss (nn.CrossEntropyLoss) ---
print("\n--- 2.1. Cross-Entropy Loss (nn.CrossEntropyLoss) ---\n")
print("The Workhorse for Language Modeling and Classification.")
print("   - How it works (conceptually for classification):")
print("     - The model outputs raw scores (logits) for each class.")
print("     - These logits are converted to probabilities (typically via Softmax).")
print("     - Cross-Entropy measures the 'distance' between the predicted probability distribution")
print("       and the true distribution (which is one-hot encoded for the correct class).")
print("   - For Language Modeling (Next-Token Prediction):")
print("     - The task is to predict the next token in a sequence.")
print("     - This is treated as a multi-class classification problem where the 'classes' are all possible tokens in the vocabulary.")
print("     - `nn.CrossEntropyLoss` is used to compare the model's predicted probability distribution over the vocabulary")
print("       for the next token against the actual next token.")
print("   - For Fine-tuning (e.g., Text Classification):")
print("     - If you adapt an LLM for sentiment analysis (e.g., positive, negative, neutral),")
print("       `nn.CrossEntropyLoss` is used to penalize incorrect classifications.")

print("\nPyTorch Example: Cross-Entropy Loss")

# --- Example 1: Multi-class Classification ---
print("\n  Example 1: Multi-class Classification")
# Model outputs (logits): Batch size 3, 5 classes
logits_clf = torch.randn(3, 5, device=device) # Raw scores for 3 samples, 5 classes each
# True labels: Batch size 3 (class indices)
targets_clf = torch.tensor([1, 0, 4], device=device) # Sample 0 is class 1, Sample 1 is class 0, etc.

loss_fn_ce_clf = nn.CrossEntropyLoss()
loss_ce_clf = loss_fn_ce_clf(logits_clf, targets_clf)

print(f"    Logits (classification): {logits_clf}")
print(f"    Targets (classification): {targets_clf}")
print(f"    Cross-Entropy Loss (classification): {loss_ce_clf.item():.4f}")

# --- Example 2: Language Modeling (Next-Token Prediction) ---
print("\n  Example 2: Language Modeling (Next-Token Prediction)")
# Model outputs (logits): Batch size 2, Sequence length 4, Vocab size 10
# For each of the 4 positions in 2 sequences, predict one of 10 tokens
vocab_size_lm = 10
seq_len_lm = 4
batch_size_lm = 2
logits_lm = torch.randn(batch_size_lm, seq_len_lm, vocab_size_lm, device=device)
# True next tokens: Batch size 2, Sequence length 4
targets_lm = torch.randint(0, vocab_size_lm, (batch_size_lm, seq_len_lm), device=device)

loss_fn_ce_lm = nn.CrossEntropyLoss() # Same loss function
# `CrossEntropyLoss` expects inputs as (N, C) or (N, C, d1, d2, ...) for K-dim loss
# and targets as (N) or (N, d1, d2, ...)
# So, we need to reshape logits to (Batch * SeqLen, VocabSize) and targets to (Batch * SeqLen)
loss_ce_lm = loss_fn_ce_lm(logits_lm.view(-1, vocab_size_lm), targets_lm.view(-1))

print(f"    Logits (LM) shape: {logits_lm.shape}")
print(f"    Targets (LM) shape: {targets_lm.shape}")
print(f"    Cross-Entropy Loss (LM): {loss_ce_lm.item():.4f}")
print("    Note: For LM, logits are reshaped to (Batch * SeqLen, VocabSize) and targets to (Batch * SeqLen).")


# --- 2.2. Negative Log-Likelihood Loss (nn.NLLLoss) ---
print("\n--- 2.2. Negative Log-Likelihood Loss (nn.NLLLoss) ---\n")
print("Closely related to Cross-Entropy Loss.")
print("`nn.CrossEntropyLoss` in PyTorch actually combines `nn.LogSoftmax()` and `nn.NLLLoss()` in one step")
print("for better efficiency and numerical stability.")
print("If your model's final layer already outputs log-probabilities (e.g., after applying `F.log_softmax` or `nn.LogSoftmax`),")
print("then you would use `nn.NLLLoss` directly.")

print("\nPyTorch Example: NLLLoss")
# Model outputs (log-probabilities): Batch size 3, 5 classes
log_probs_nll = F.log_softmax(logits_clf, dim=1) # Using logits from previous example
# True labels: Batch size 3 (class indices) - same targets_clf

loss_fn_nll = nn.NLLLoss()
loss_nll = loss_fn_nll(log_probs_nll, targets_clf)

print(f"    Logits (from CE example): \n{logits_clf}")
print(f"    Log Probs (after F.log_softmax(logits)): \n{log_probs_nll}")
print(f"    Targets: {targets_clf}")
print(f"    NLL Loss: {loss_nll.item():.4f}")
print(f"    (Compare NLL Loss with previous Cross-Entropy Loss: {loss_ce_clf.item():.4f} - they should be the same)")


# --- 2.3. Mean Squared Error Loss (nn.MSELoss) ---
print("\n--- 2.3. Mean Squared Error Loss (nn.MSELoss) ---\n")
print("Used for Regression Tasks.")
print("Calculates the average squared difference between predicted continuous values and actual continuous values.")
print("Less common for core LLM pre-training but can be used if an LLM is fine-tuned for a task")
print("that requires predicting a numerical value (e.g., predicting a review score from 1.0 to 5.0).")

print("\nPyTorch Example: MSELoss")
# Model predictions (continuous values): Batch size 4
predictions_mse = torch.randn(4, 1, device=device) * 5 # e.g., predicted scores
# True values (continuous): Batch size 4
targets_mse = predictions_mse + 0.1 # e.g., simulated actual scores

loss_fn_mse = nn.MSELoss()
loss_mse = loss_fn_mse(predictions_mse, targets_mse)

print(f"    Predictions (MSE): \n{predictions_mse.squeeze()}")
print(f"    Targets (MSE): \n{targets_mse.squeeze()}")
print(f"    MSE Loss: {loss_mse.item():.4f}")


# --- 2.4. KL Divergence Loss (nn.KLDivLoss) ---
print("\n--- 2.4. KL Divergence Loss (nn.KLDivLoss) ---\n")
print("Kullback-Leibler (KL) Divergence measures how one probability distribution P diverges from a second,")
print("expected probability distribution Q. It's a measure of the 'information lost' when Q is used to approximate P.")
print("`nn.KLDivLoss` expects the input to be log-probabilities and the target to be probabilities.")
print("   - Use Cases in LLMs:")
print("     - Knowledge Distillation: Training a smaller 'student' model to mimic the output probability")
print("       distribution (softmax outputs) of a larger 'teacher' model. The KL Divergence between the")
print("       student's and teacher's softmax outputs (student's as log-probs, teacher's as probs) can be a loss component.")
print("     - Aligning Distributions: Ensuring the model's output distribution has certain desired properties")
print("       or matches a target distribution (e.g., in Variational Autoencoders - VAEs).")

print("\nPyTorch Example: KLDivLoss")
# Example: Aligning a predicted distribution with a target distribution.
# Input (log-probabilities from model): Batch size 2, 4 classes
input_log_probs_kld = F.log_softmax(torch.randn(2, 4, device=device), dim=1)
# Target (probabilities): Batch size 2, 4 classes
# Target probabilities must sum to 1 for each sample in the batch.
target_probs_kld = F.softmax(torch.rand(2, 4, device=device), dim=1) # Create valid probability distributions

loss_fn_kld = nn.KLDivLoss(reduction='batchmean') # 'batchmean' averages over batch dimension
# Other reductions: 'mean' (average over all elements), 'sum'
loss_kld = loss_fn_kld(input_log_probs_kld, target_probs_kld)

print(f"    Input (Log Probs for KLD): \n{input_log_probs_kld}")
print(f"    Target (Probs for KLD): \n{target_probs_kld}")
print(f"    KL Divergence Loss: {loss_kld.item():.4f}")

print("\n  Note on KLDivLoss for Knowledge Distillation:")
print("  `Loss = alpha * T^2 * KLDivLoss(log_softmax(student_logits/T), softmax(teacher_logits/T))`")
print("  Where T is temperature, alpha is a weighting factor. The T^2 term scales the gradient.")


# --- 2.5. Specialized Loss Functions (Conceptual Mentions) ---
print("\n--- 2.5. Specialized Loss Functions (Conceptual Mentions) ---\n")
print("   - Contrastive Loss (e.g., MarginRankingLoss, TripletMarginLoss):")
print("     - Used to learn embeddings where similar items are pulled closer and dissimilar items are pushed apart in the embedding space.")
print("     - Example: Training sentence encoders where paraphrases should have similar embeddings, while unrelated sentences should have dissimilar ones.")
print("     - `nn.TripletMarginLoss` aims to ensure that an anchor sample is closer to a positive sample than to a negative sample by at least a margin.")
print("   - Ranking Loss:")
print("     - For tasks where the relative order of items is important (e.g., learning to rank search results, document relevance).")
print("     - `nn.MarginRankingLoss` can be used here too, comparing pairs of items.")


# %% 3. Combining Loss Functions
print("\n--- 3. Combining Loss Functions ---\n")

print("In many advanced scenarios, especially in multi-task learning or when using auxiliary (secondary) losses,")
print("you might combine multiple loss functions.")

print("\nWhy Combine Loss Functions?")
print("1. Multi-Task Learning (MTL):")
print("   - Training a single model to perform several tasks simultaneously.")
print("   - Example: An LLM fine-tuned to both classify sentiment AND predict the emotion intensity (regression).")
print("     Each task would have its own loss component (e.g., CrossEntropy for sentiment, MSE for intensity).")
print("   - Can lead to better generalization as the model learns more robust representations by leveraging shared information across tasks.")
print("2. Auxiliary Losses for Regularization or Improved Representation Learning:")
print("   - An auxiliary (secondary) loss can guide the main task by encouraging the model to learn useful intermediate representations or adhere to certain constraints.")
print("   - Example: In addition to next-token prediction, an LLM might have an auxiliary loss that")
print("     predicts sentence coherence or some other linguistic property to improve the quality of generated text.")
print("3. Knowledge Distillation (as mentioned with KL Divergence):")
print("   - `Loss_Total = Loss_CrossEntropy_on_Hard_Labels + alpha * Loss_KL_Divergence_with_Teacher_Soft_Labels`")
print("     Here, the cross-entropy loss trains the student on the ground truth labels, while the KL divergence loss")
print("     encourages the student to mimic the teacher's softened probability outputs.")

print("\nHow to Combine Loss Functions?")
print("The most common method is a weighted sum:")
print("`Total Loss = w1 * Loss1 + w2 * Loss2 + ... + wn * Lossn`")
print("- `Loss1`, `Loss2`, etc., are individual loss values from different tasks or components.")
print("- `w1`, `w2`, etc., are scalar weights that determine the contribution of each loss component.")
print("  - These weights are hyperparameters and are crucial. They often require careful tuning and experimentation.")
print("  - Poorly chosen weights can lead to one task dominating others or instability in training.")
print("  - Weights can be static (fixed throughout training) or dynamic (e.g., changing based on task progress or uncertainty).")

print("\nPyTorch Example: Combining Losses for a Mock Multi-Task Scenario")
# --- Conceptual Example of Combined Loss in PyTorch ---
# Imagine a model that performs two tasks:
# Task 1: Classification (e.g., sentiment)
# Task 2: Regression (e.g., intensity of sentiment)

# Mock model outputs for a batch of 3 samples
batch_size_multi = 3
num_classes_multi = 3 # For classification task
# Output for Task 1 (classification logits)
output_task1_logits = torch.randn(batch_size_multi, num_classes_multi, device=device)
# Output for Task 2 (regression values)
output_task2_values = torch.randn(batch_size_multi, 1, device=device)

# Mock targets
targets_task1_labels = torch.tensor([0, 2, 1], device=device) # Class labels
targets_task2_values = torch.tensor([[0.8], [0.2], [0.5]], device=device) # Regression targets

# Loss functions for each task
loss_fn_task1 = nn.CrossEntropyLoss()
loss_fn_task2 = nn.MSELoss()

# Calculate individual losses
loss1_val = loss_fn_task1(output_task1_logits, targets_task1_labels)
loss2_val = loss_fn_task2(output_task2_values, targets_task2_values)

# Weights for combining losses
weight1 = 0.7 # Weight for classification task
weight2 = 0.3 # Weight for regression task

# Combine losses
total_loss = weight1 * loss1_val + weight2 * loss2_val

print(f"  Output Task 1 (Logits): \n{output_task1_logits}")
print(f"  Targets Task 1 (Labels): {targets_task1_labels}")
print(f"  Loss Task 1 (CE): {loss1_val.item():.4f}")

print(f"\n  Output Task 2 (Values): \n{output_task2_values.squeeze()}")
print(f"  Targets Task 2 (Values): {targets_task2_values.squeeze()}")
print(f"  Loss Task 2 (MSE): {loss2_val.item():.4f}")

print(f"\n  Weight for Task 1 Loss: {weight1}")
print(f"  Weight for Task 2 Loss: {weight2}")
print(f"  Combined Total Loss: ({weight1} * {loss1_val.item():.4f}) + ({weight2} * {loss2_val.item():.4f}) = {total_loss.item():.4f}")

# In a real training loop, you would then call:
# optimizer.zero_grad()
# total_loss.backward() # Gradients are backpropagated based on the combined loss
# optimizer.step()

print("\nConsiderations when combining losses:")
print("- Scale of Losses: Individual losses might be on vastly different scales. This can cause")
print("  one loss to dominate the gradient updates. Normalizing them (e.g., by dividing by their")
print("  initial values or running averages) or carefully choosing weights is important.")
print("- Gradient Conflicts: In multi-task learning, optimizing for one task might sometimes hurt")
print("  performance on another if the tasks are conflicting (negative transfer). Advanced techniques")
print("  exist to manage this (e.g., gradient normalization, uncertainty weighting).")
print("- Dynamic Weighting Schemes: Some methods dynamically adjust weights during training, for example,")
print("  based on task uncertainty, learning progress, or the magnitude of gradients for each task.")


# %% 4. Summary
print("\n--- 4. Summary ---\n")
print("Understanding and choosing appropriate loss functions is fundamental to successfully training LLMs.")
print("- Cross-Entropy Loss is a staple for both language modeling and classification tasks.")
print("- Other losses like MSE (for regression) and KL Divergence (for distribution matching or distillation)")
print("  serve specialized roles when fine-tuning or extending LLMs.")
print("- Combining losses through weighted sums allows models to learn from multiple objectives simultaneously,")
print("  which is key for multi-task learning and incorporating auxiliary objectives.")
print("Effective use of loss functions, including their combination, often involves experimentation and tuning")
print("to achieve the best performance on the desired downstream tasks.")

print("\nEnd of Module 14.")
