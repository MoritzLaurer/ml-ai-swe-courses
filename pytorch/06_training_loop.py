# -*- coding: utf-8 -*-
# 06_training_loop.py

# Module 6: The PyTorch Training Loop: Optimizers, Loss Functions & Evaluation
#
# This script covers:
# 1. Setting up: Model, DataLoaders, Loss, Optimizer, LR Scheduler.
# 2. The Standard Training Loop Structure (Epochs & Batches).
# 3. Core Training Steps: zero_grad, forward, loss, backward, step.
# 4. Learning Rate Scheduling (`lr_scheduler`).
# 5. The Evaluation Loop: Using `model.eval()` and `torch.no_grad()`.
# 6. Calculating Metrics: Perplexity from Cross-Entropy Loss.
# 7. Putting it Together: A basic training function.
# 8. Performance Monitoring Notes.
# 9. Comparisons to JAX training paradigms.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import math
import time

print("--- Module 6: The PyTorch Training Loop ---\n")
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


# %% 1. Setting Up: Model, Data, Loss, Optimizer, Scheduler
print("\n--- 1. Setting Up: Model, Data, Loss, Optimizer, Scheduler ---\n")

# --- Dummy Data and Dataset (Simple Sequence Task) ---
# Let's create a task: given a sequence of numbers, predict the next number.
# Example: input [1, 2, 3], target [2, 3, 4]
SEQ_LEN = 20
VOCAB_SIZE = 40 # Number of unique "tokens" (numbers in this case)
PAD_IDX = 0     # Use 0 for padding index

# Very simple dummy data where target is just the input sequence shifted by one to the right
# (maintaining the same length)
def create_dummy_sequence_data(num_samples, max_val):
    data = []
    for _ in range(num_samples):
        start = torch.randint(1, max_val - SEQ_LEN, (1,)).item() # Ensure space for sequence
        seq = torch.arange(start, start + SEQ_LEN + 1) # Create sequence + next item
        input_seq = seq[:-1] # Input sequence
        target_seq = seq[1:]  # Target sequence (shifted by one)
        data.append((input_seq, target_seq))
    return data

train_data_raw = create_dummy_sequence_data(200, VOCAB_SIZE)
val_data_raw = create_dummy_sequence_data(40, VOCAB_SIZE)

print(f"Train data: {train_data_raw[:5]}")

class SequenceDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # Return as tensors
        in_seq, tgt_seq = self.data[idx]
        return {"input_ids": in_seq.long(), "target_ids": tgt_seq.long()}

train_dataset = SequenceDataset(train_data_raw)
val_dataset = SequenceDataset(val_data_raw)

# Collate function (no padding needed if sequences are fixed length)
def simple_collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    target_ids = torch.stack([item['target_ids'] for item in batch])
    return {"input_ids": input_ids, "target_ids": target_ids}

BATCH_SIZE = 16
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=simple_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=simple_collate_fn)
print(f"Created DataLoaders. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# --- Simple Model (Embedding -> Linear) ---
class SimpleSeqModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        # We could add an RNN/LSTM here, but keeping it simple for focus on training loop
        # self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(embed_dim, hidden_dim) # Simplified: just linear on embedding
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, vocab_size) # Predict next token ID

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        embedded = self.embedding(input_ids) # -> (batch_size, seq_len, embed_dim)
        # If using RNN: hidden_states, _ = self.rnn(embedded) # -> (batch_size, seq_len, hidden_dim)
        hidden_states = self.relu(self.linear(embedded)) # Simplified # -> (batch_size, seq_len, hidden_dim)
        logits = self.output_layer(hidden_states) # -> (batch_size, seq_len, vocab_size)
        return logits

EMBED_DIM = 32
HIDDEN_DIM = 64
model = SimpleSeqModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM).to(device)
print(f"Created Model:\n{model}")

# --- Loss Function ---
# For next-token prediction (classification over vocab), CrossEntropyLoss is standard.
# It expects model output (logits) of shape (N, C) or (N, C, d1, d2...)
# and target of shape (N) or (N, d1, d2...) where C is the number of classes (vocab_size).
# Our model output is (batch, seq_len, vocab_size), target is (batch, seq_len).
# We need to reshape them to (batch * seq_len, vocab_size) and (batch * seq_len).
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX) # Ignore padding index in loss calculation
print(f"Using Loss Function: {loss_fn}")

# --- Optimizer ---
# AdamW is often preferred for Transformer-based models, but Adam/SGD work too.
LEARNING_RATE = 1e-3
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
print(f"Using Optimizer: {optimizer}")

# --- Learning Rate Scheduler ---
# Adjusts the learning rate during training (e.g., decrease over time).
# StepLR decreases LR by `gamma` every `step_size` epochs.
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) # Decrease LR by 10x every 5 epochs
print(f"Using LR Scheduler: {scheduler.__class__.__name__}")


# %% 2. The Standard Training Loop Structure
print("\n--- 2. The Standard Training Loop Structure ---\n")

print("""
Typical PyTorch training involves nested loops:
- Outer loop: Iterates over epochs (passes through the entire dataset).
- Inner loop: Iterates over batches provided by the DataLoader.
""")

# Pseudocode:
# for epoch in range(num_epochs):
#     model.train() # Set model to training mode
#     for batch in train_loader:
#         # --- Core Training Steps ---
#         pass # Details below
#
#     model.eval() # Set model to evaluation mode
#     with torch.no_grad():
#         for batch in val_loader:
#             # --- Evaluation Steps ---
#             pass # Details below
#
#     scheduler.step() # Update learning rate (usually per epoch)


# %% 3. Core Training Steps (Inside the Inner Loop)
print("\n--- 3. Core Training Steps (Inside the Inner Loop) ---\n")

print("""
For each batch during training:
1. `optimizer.zero_grad()`: Clear gradients from the previous step. Crucial!
2. Move data to the correct device: `inputs = batch['input_ids'].to(device)` etc.
3. Forward pass: `outputs = model(inputs)` to get model predictions (logits).
4. Calculate loss: `loss = loss_fn(outputs.view(-1, VOCAB_SIZE), targets.view(-1))` (Reshape for CrossEntropyLoss).
5. Backward pass: `loss.backward()` calculates gradients of the loss w.r.t. model parameters.
6. Optimizer step: `optimizer.step()` updates model parameters using the calculated gradients.
""")


# %% 4. Learning Rate Scheduling
print("\n--- 4. Learning Rate Scheduling ---\n")

print("LR Schedulers adjust the learning rate based on defined rules.")
print("Common practice: call `scheduler.step()` once per epoch, after the training loop for that epoch.")
print(f"Initial LR: {optimizer.param_groups[0]['lr']}")
# Simulate a few scheduler steps
# scheduler.step()
# print(f"LR after 1 epoch: {optimizer.param_groups[0]['lr']}")
# ... after step_size epochs, LR should decrease


# %% 5. The Evaluation Loop
print("\n--- 5. The Evaluation Loop ---\n")

print("""
Evaluation assesses model performance on unseen data (validation set).
Key elements:
1. `model.eval()`: Sets the model to evaluation mode. This disables Dropout layers
   and makes Batch Normalization layers use running statistics instead of batch statistics. Essential for reproducibility.
2. `with torch.no_grad()`: Disables gradient calculation, saving memory and computation,
   as gradients are not needed during evaluation.
3. Iterate through `val_loader`.
4. Perform forward pass and calculate loss/metrics.
5. Aggregate metrics over all validation batches.
6. Remember `model.train()`: Switch back to training mode after evaluation.
""")


# %% 6. Calculating Metrics: Perplexity
print("\n--- 6. Calculating Metrics: Perplexity ---\n")

print("Perplexity (PPL) is a common metric for language models.")
print("It measures how well a probability model predicts a sample.")
print("Lower Perplexity is better.")
print("Formula: PPL = exp(average cross-entropy loss)")
print("Calculation steps in evaluation loop:")
print(" - Calculate cross-entropy loss for each token (ignoring padding).")
print(" - Compute the average loss across all tokens in the validation set.")
print(" - Calculate `math.exp(average_loss)`.")


# %% 7. Putting It Together: Training Function
print("\n--- 7. Putting It Together: Training Function ---\n")

def evaluate(model, dataloader, loss_fn, device, vocab_size, pad_idx):
    """Evaluates the model on the given dataloader."""
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad(): # Disable gradient calculation
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device) # Shape: (batch, seq_len)

            # Forward pass
            logits = model(input_ids) # Shape: (batch, seq_len, vocab_size)

            # Reshape for CrossEntropyLoss
            # Logits: (batch * seq_len, vocab_size)
            # Target: (batch * seq_len)
            loss = loss_fn(logits.view(-1, vocab_size), target_ids.view(-1))

            # Calculate number of non-padding tokens in target for accurate averaging
            # Create a mask for non-padding tokens
            non_pad_mask = (target_ids.view(-1) != pad_idx)
            num_tokens = non_pad_mask.sum().item()

            # Accumulate total loss, weighted by the number of tokens
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    if total_tokens == 0: 
        return 0.0, float('inf') # Avoid division by zero

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def train_model(model, train_loader, val_loader, loss_fn, optimizer, scheduler,
                num_epochs, device, vocab_size, pad_idx):
    """Runs the training and evaluation loops."""
    start_time_all = time.time()
    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

        # --- Training Phase ---
        model.train() # Set model to training mode
        epoch_train_loss = 0.0
        processed_batches = 0

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad() # 1. Clear gradients

            input_ids = batch['input_ids'].to(device) # 2. Move data
            target_ids = batch['target_ids'].to(device)

            logits = model(input_ids) # 3. Forward pass

            # 4. Calculate loss (Reshape for CrossEntropyLoss)
            loss = loss_fn(logits.view(-1, vocab_size), target_ids.view(-1))

            loss.backward() # 5. Backward pass

            # Optional: Gradient Clipping (uncomment if needed)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step() # 6. Optimizer step

            epoch_train_loss += loss.item()
            processed_batches += 1

            # Optional: Log batch loss periodically
            if (i + 1) % max(1, len(train_loader)//2) == 0: # Log twice per epoch
                 print(f"  Batch {i+1}/{len(train_loader)}, Train Loss: {loss.item():.4f}")

        avg_epoch_train_loss = epoch_train_loss / processed_batches
        print(f"Epoch {epoch+1} Training Average Loss: {avg_epoch_train_loss:.4f}")

        # --- Evaluation Phase ---
        avg_val_loss, perplexity = evaluate(model, val_loader, loss_fn, device, vocab_size, pad_idx)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}, Perplexity: {perplexity:.4f}")

        # --- LR Scheduler Step ---
        scheduler.step()
        print(f"Epoch {epoch+1} finished. New LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Time taken: {time.time() - epoch_start_time:.2f} seconds")


    total_time = time.time() - start_time_all
    print(f"\nTraining finished after {total_time:.2f} seconds.")


# --- Run Training ---
NUM_EPOCHS = 5 # Run for a few epochs for demonstration
train_model(model, train_loader, val_loader, loss_fn, optimizer, scheduler,
            NUM_EPOCHS, device, VOCAB_SIZE, PAD_IDX)


# %% 8. Performance Monitoring Notes
print("\n--- 8. Performance Monitoring Notes ---\n")

print("* During training (especially on GPU), monitor resource usage:")
print("    - `nvidia-smi` (in terminal): Shows GPU utilization, memory usage, temperature.")
print("    - Check CPU utilization (e.g., `htop` on Linux): High CPU might indicate data loading bottlenecks (`num_workers`).")
print("* For detailed bottleneck analysis, use PyTorch Profiler:")
print("    - `with torch.profiler.profile(...) as prof:` context manager.")
print("    - Can show time spent in CPU/GPU operations, data loading, etc.")
print("    - Helps identify specific slow parts of your code.")


# %% 9. JAX Comparison
print("\n--- 9. JAX Comparison ---\n")

print("Training in JAX often looks different:")
print("- Training Step as a Function: The core logic (forward, loss, grads, update) is typically encapsulated in a single Python function.")
print("- JIT Compilation: This function is then compiled using `@jax.jit` for performance.")
print("- Functional Gradients: Gradients are computed using `jax.value_and_grad()` applied to a loss function.")
print("- Optimizer State: Optimizers (like Optax) are functional. They don't modify parameters in-place.")
print("  Instead, the optimizer takes current params and gradients, and returns *new* updated params and a *new* optimizer state.")
print("  This state (e.g., momentum vectors) must be explicitly passed around.")
print("- Loop Structure: The Python loop iterates, calling the `@jit`-compiled step function, passing in and receiving updated parameters and optimizer state.")
print("- LR Scheduling: Often implemented as functions that compute the LR based on step count, passed into the optimizer update.")
print("- Evaluation: Similar concepts (`no_grad`-like context often not needed as grad calculation is explicit), but implementation uses JAX functions/arrays.")
print("\nPyTorch feels more object-oriented and stateful, while JAX feels more functional and explicit about state management.")


# %% Conclusion
print("\n--- Module 6 Summary ---\n")
print("Key Takeaways:")
print("- The PyTorch training loop involves iterating epochs and batches.")
print("- Key steps per batch: zero_grad -> forward -> loss -> backward -> step.")
print("- `nn.CrossEntropyLoss` is common for classification/LM, requires specific input shapes.")
print("- Optimizers (`optim.AdamW`) update parameters; Schedulers (`lr_scheduler`) adjust LR.")
print("- Evaluation requires `model.eval()`, `torch.no_grad()`, and appropriate metrics (e.g., Perplexity).")
print("- Separate train/validation loops are standard practice.")
print("- PyTorch's stateful optimizers/schedulers contrast with JAX's functional approach.")

print("\nEnd of Module 6.")