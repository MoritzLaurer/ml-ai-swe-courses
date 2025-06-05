# -*- coding: utf-8 -*-
# 15_multi_task_llm_fine_tuning.py

# Module 15: Multi-Task Fine-Tuning of LLMs (SFT + Regression)
#
# This script demonstrates how to modify a pre-trained autoregressive LLM
# to simultaneously fine-tune it for:
# 1. Supervised Fine-Tuning (SFT - continued next-token prediction).
# 2. A regression task (e.g., acting as an LLM judge to output a score).
#
# We will cover:
# - Defining a multi-task model architecture with separate heads.
# - Preparing data for both tasks.
# - Calculating and combining losses from both heads.
# - A conceptual training loop.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

print("--- Module 15: Multi-Task Fine-Tuning of LLMs (SFT + Regression) ---\n")
print(f"Using PyTorch version: {torch.__version__}")


# --- Reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

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


# %% 1. Conceptual Pre-trained Backbone (Simplified SmallGPT)
#    We'll use a simplified version of the SmallGPT model from Module 8
#    as our pre-trained backbone.

class TransformerBlock(nn.Module): # Simplified from Module 8
    def __init__(self, embed_dim, num_heads, ff_dim_multiplier=4, dropout_rate=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        ff_dim = embed_dim * ff_dim_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.ReLU(),
            nn.Linear(ff_dim, embed_dim), nn.Dropout(dropout_rate)
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, causal_mask=None): # Added causal_mask argument
        _batch_size, seq_len, _embed_dim = x.shape
        if causal_mask is None: # Default to causal if not provided
             causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                sz=seq_len, device=x.device, dtype=torch.bool # PyTorch >= 1.9 expects bool
            )
        else: # Ensure mask is boolean if provided
            causal_mask = causal_mask.to(dtype=torch.bool)

        norm_x_attn = self.ln1(x)
        attn_output, _ = self.attn(norm_x_attn, norm_x_attn, norm_x_attn,
                                   attn_mask=causal_mask, is_causal=False, need_weights=False)
        x = x + self.dropout(attn_output)
        norm_x_ffn = self.ln2(x)
        ff_output = self.ffn(norm_x_ffn)
        x = x + ff_output
        return x

class PretrainedBackbone(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, block_size, dropout_rate):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(block_size, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, dropout_rate=dropout_rate) for _ in range(num_layers)]
        )
        self.ln_final = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size) # Original LM head

        # Weight tying
        self.token_embedding.weight = self.lm_head.weight
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1 and 'embedding' not in name :
                 nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                 nn.init.zeros_(param)
            elif 'embedding.weight' in name:
                 nn.init.normal_(param, mean=0.0, std=0.02)

    def forward(self, idx_sequences, attention_mask=None):
        # idx_sequences shape: (batch_size, seq_len)
        batch_size, seq_len = idx_sequences.shape
        tok_emb = self.token_embedding(idx_sequences)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=idx_sequences.device)
        pos_emb = self.positional_embedding(positions)
        x = tok_emb + pos_emb
        x = self.dropout(x)

        # Create causal mask if not provided (standard for autoregressive)
        # Or use provided attention_mask (e.g., for padding)
        causal_mask = None
        if attention_mask is not None:
            # For MultiheadAttention, a True value indicates that the corresponding position is not allowed to attend.
            # We need to invert the HF-style attention_mask (1 for attend, 0 for not attend)
            # and also make it suitable for causal masking if needed.
            # This part can be complex depending on how padding and causality interact.
            # For simplicity here, we'll assume SFT uses standard causal masking implicitly.
            # And for regression, we might pool over non-padded tokens.
            # Let's assume for now standard causal masking is handled by blocks if attention_mask is None.
            pass # Simplified for this example

        for block in self.transformer_blocks:
            x = block(x, causal_mask=None) # Pass causal mask to blocks

        hidden_states = self.ln_final(x) # Shape: (batch, seq_len, embed_dim)
        lm_logits = self.lm_head(hidden_states) # Shape: (batch, seq_len, vocab_size)
        return lm_logits, hidden_states


# %% 2. Multi-Task LLM Definition
print("\n--- 2. Multi-Task LLM Definition ---\n")

class MultiTaskLLM(nn.Module):
    def __init__(self, backbone_model, num_regression_outputs=1, regression_pool_type='last'):
        super().__init__()
        self.backbone = backbone_model
        self.embed_dim = backbone_model.lm_head.in_features # Get embed_dim from backbone
        self.regression_pool_type = regression_pool_type

        # New Regression Head
        # Takes the pooled hidden state and outputs regression value(s)
        self.regression_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1), # Optional dropout for regression head
            nn.Linear(self.embed_dim // 2, num_regression_outputs)
        )
        print(f"MultiTaskLLM initialized with a regression head for {num_regression_outputs} output(s).")
        print(f"Regression pooling type: {regression_pool_type}")


    def forward(self, idx_sequences, attention_mask=None, task="all"):
        """
        Args:
            idx_sequences (Tensor): Input token IDs (batch, seq_len)
            attention_mask (Tensor, optional): Mask to avoid performing attention on padding token indices.
                                               (batch, seq_len), 1 for tokens to attend, 0 for padding.
            task (str): "sft", "regression", or "all". Determines which outputs to compute.
        Returns:
            dict: A dictionary containing 'lm_logits' and/or 'regression_output'.
        """
        # Get outputs from the backbone
        # lm_logits: (batch, seq_len, vocab_size)
        # hidden_states: (batch, seq_len, embed_dim)
        lm_logits, hidden_states = self.backbone(idx_sequences, attention_mask=attention_mask)

        outputs = {}

        if task in ["sft", "all"]:
            outputs['lm_logits'] = lm_logits

        if task in ["regression", "all"]:
            # Pool hidden states for regression task
            if self.regression_pool_type == 'last':
                # Use the hidden state of the last token
                # This requires careful handling of padding if sequences have variable lengths.
                # If using an attention_mask, find the last non-padded token.
                if attention_mask is not None:
                    # Get the length of each sequence in the batch from the attention mask
                    sequence_lengths = attention_mask.sum(dim=1) - 1 # 0-indexed
                    # Gather the hidden states of the last actual token for each sequence
                    pooled_hidden_state = hidden_states[torch.arange(hidden_states.size(0), device=device), sequence_lengths]
                else: # Assume no padding, use the very last token
                    pooled_hidden_state = hidden_states[:, -1, :] # (batch, embed_dim)
            elif self.regression_pool_type == 'mean':
                # Mean pooling over non-padded token hidden states
                if attention_mask is not None:
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_hidden_states = torch.sum(hidden_states * input_mask_expanded, dim=1)
                    sum_mask = input_mask_expanded.sum(dim=1)
                    sum_mask = torch.clamp(sum_mask, min=1e-9) # Avoid division by zero
                    pooled_hidden_state = sum_hidden_states / sum_mask
                else: # Assume no padding, mean pool all
                    pooled_hidden_state = torch.mean(hidden_states, dim=1) # (batch, embed_dim)
            else:
                raise ValueError(f"Unsupported regression_pool_type: {self.regression_pool_type}")

            regression_output = self.regression_head(pooled_hidden_state) # (batch, num_regression_outputs)
            outputs['regression_output'] = regression_output

        return outputs

# --- Configuration ---
MODEL_CONFIG_BACKBONE = {
    'vocab_size': 1000,  # Small vocab for example
    'embed_dim': 64,
    'num_heads': 4,
    'num_layers': 3,
    'block_size': 32,   # Max sequence length
    'dropout_rate': 0.1
}

# --- Instantiate ---
# 1. Create a dummy pre-trained backbone
print("\nInstantiating dummy pre-trained backbone...")
backbone = PretrainedBackbone(**MODEL_CONFIG_BACKBONE).to(device)
# In a real scenario, you would load pre-trained weights into the backbone here.
# e.g., backbone.load_state_dict(torch.load("path/to/pretrained_weights.pth"))

# 2. Create the Multi-Task LLM
print("Instantiating MultiTaskLLM...")
multi_task_model = MultiTaskLLM(backbone, num_regression_outputs=1, regression_pool_type='mean').to(device)

# --- Test Forward Pass ---
print("\nTesting forward pass...")
batch_size_test = 4
seq_len_test = MODEL_CONFIG_BACKBONE['block_size']
dummy_input_ids = torch.randint(0, MODEL_CONFIG_BACKBONE['vocab_size'], (batch_size_test, seq_len_test), device=device)
# Dummy attention mask (e.g., some sequences are padded)
dummy_attention_mask = torch.ones_like(dummy_input_ids)
dummy_attention_mask[0, -5:] = 0 # First sample has last 5 tokens padded
dummy_attention_mask[1, -10:] = 0 # Second sample has last 10 tokens padded


print(f"  Dummy input shape: {dummy_input_ids.shape}")
print(f"  Dummy attention mask (first sample): {dummy_attention_mask[0]}")

# Test getting all outputs
all_outputs = multi_task_model(dummy_input_ids, attention_mask=dummy_attention_mask, task="all")
if 'lm_logits' in all_outputs:
    print(f"  LM Logits output shape: {all_outputs['lm_logits'].shape}")
if 'regression_output' in all_outputs:
    print(f"  Regression output shape: {all_outputs['regression_output'].shape}")

# Test getting only SFT output
sft_output = multi_task_model(dummy_input_ids, attention_mask=dummy_attention_mask, task="sft")
if 'lm_logits' in sft_output and 'regression_output' not in sft_output:
    print("  Successfully got SFT-only output.")

# Test getting only regression output
regression_output = multi_task_model(dummy_input_ids, attention_mask=dummy_attention_mask, task="regression")
if 'regression_output' in regression_output and 'lm_logits' not in regression_output:
     print("  Successfully got regression-only output.")


# %% 3. Data Preparation (Conceptual)
print("\n--- 3. Data Preparation (Conceptual) ---\n")
print("For multi-task training, your dataset needs to provide inputs and targets for ALL tasks.")
print("Each data sample would typically contain:")
print("  - `input_ids`: The tokenized input sequence.")
print("  - `attention_mask`: To handle padding.")
print("  - `lm_labels` (or `target_ids`): The target token IDs for next-token prediction (SFT).")
print("    Often, these are the `input_ids` shifted, with padding tokens ignored in the loss.")
print("  - `regression_target`: The target numerical value for the regression task.")

# --- Dummy MultiTaskDataset ---
class DummyMultiTaskDataset(Dataset):
    def __init__(self, num_samples, vocab_size, max_seq_len):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq_len = random.randint(self.max_seq_len // 2, self.max_seq_len) # Variable sequence length
        input_ids = torch.randint(0, self.vocab_size, (seq_len,))
        
        # Pad sequence to max_seq_len
        padding_length = self.max_seq_len - seq_len
        padded_input_ids = torch.cat([
            input_ids,
            torch.zeros(padding_length, dtype=torch.long) # Assuming 0 is pad_token_id
        ], dim=0)

        attention_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
        attention_mask[:seq_len] = 1 # 1 for actual tokens, 0 for padding

        # SFT labels: input_ids shifted, ignore padding by setting labels to -100 (common practice)
        lm_labels = padded_input_ids.clone()
        # For SFT, typically predict the next token, so labels are shifted.
        # However, CrossEntropyLoss handles this if inputs are (N,C) and targets are (N).
        # Here, lm_labels will be the same as input_ids but with -100 for padding.
        # When computing loss, we'll shift logits and labels.
        lm_labels[attention_mask == 0] = -100 # Ignore padding in loss calculation

        # Regression target: A random float
        regression_target = torch.randn(1).float()

        return {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "lm_labels": lm_labels,
            "regression_target": regression_target
        }

# Instantiate dataset and dataloader
dummy_dataset = DummyMultiTaskDataset(num_samples=100, vocab_size=MODEL_CONFIG_BACKBONE['vocab_size'], max_seq_len=MODEL_CONFIG_BACKBONE['block_size'])
dummy_dataloader = DataLoader(dummy_dataset, batch_size=batch_size_test, shuffle=True)

print(f"\nCreated dummy dataset with {len(dummy_dataset)} samples.")
sample_batch = next(iter(dummy_dataloader))
print("Sample batch keys:", sample_batch.keys())
print("  input_ids shape:", sample_batch['input_ids'].shape)
print("  attention_mask shape:", sample_batch['attention_mask'].shape)
print("  lm_labels shape:", sample_batch['lm_labels'].shape)
print("  regression_target shape:", sample_batch['regression_target'].shape)
print(f"  Example lm_labels for one sample (padded parts are -100):\n  {sample_batch['lm_labels'][0]}")
print(f"  Example regression_target for one sample:\n  {sample_batch['regression_target'][0]}")


# %% 4. Loss Calculation and Combination
print("\n--- 4. Loss Calculation and Combination ---\n")

loss_fn_sft = nn.CrossEntropyLoss(ignore_index=-100) # -100 is often used to ignore tokens in loss
loss_fn_regression = nn.MSELoss() # Mean Squared Error for regression

# Weights for combining losses (these are hyperparameters)
sft_loss_weight = 0.7
regression_loss_weight = 0.3

print(f"SFT Loss Weight: {sft_loss_weight}, Regression Loss Weight: {regression_loss_weight}")

# --- Conceptual Training Step ---
def conceptual_train_step(batch, model, optimizer):
    model.train()
    optimizer.zero_grad()

    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    lm_labels = batch['lm_labels'].to(device)
    regression_targets = batch['regression_target'].to(device)

    # Forward pass to get all outputs
    outputs = model(input_ids, attention_mask=attention_mask, task="all")
    lm_logits = outputs['lm_logits']
    regression_predictions = outputs['regression_output']

    # 1. Calculate SFT Loss (Next-Token Prediction)
    # lm_logits: (batch, seq_len, vocab_size)
    # lm_labels: (batch, seq_len)
    # We need to align lm_logits with lm_labels.
    # Typically, for Causal LM, the logits at position `i` predict token at `i+1`.
    # So, we shift logits and labels.
    #   shift_logits = lm_logits[..., :-1, :].contiguous()
    #   shift_labels = lm_labels[..., 1:].contiguous()
    # Or, ensure labels are set up such that `lm_labels[i]` is the target for `input_ids[i]`.
    # For CrossEntropyLoss, if `lm_logits` is (N, C) and `lm_labels` is (N):
    sft_loss = loss_fn_sft(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))

    # 2. Calculate Regression Loss
    regression_loss = loss_fn_regression(regression_predictions, regression_targets)

    # 3. Combine Losses
    total_loss = (sft_loss_weight * sft_loss) + \
                 (regression_loss_weight * regression_loss)

    # Backward pass and optimization
    total_loss.backward()
    optimizer.step()

    return total_loss.item(), sft_loss.item(), regression_loss.item()


# %% 5. Conceptual Training Loop
print("\n--- 5. Conceptual Training Loop ---\n")

# Optimizer (AdamW is common for LLMs)
# You might want differential learning rates: smaller for backbone, larger for new heads.
# For simplicity, one learning rate for now.
optimizer = optim.AdamW(multi_task_model.parameters(), lr=1e-4)

num_epochs_example = 3 # Small number for demo
print(f"Starting conceptual training for {num_epochs_example} epochs...")

for epoch in range(num_epochs_example):
    epoch_total_loss = 0
    epoch_sft_loss = 0
    epoch_reg_loss = 0
    num_batches = 0

    for batch in dummy_dataloader:
        total_loss, sft_loss, reg_loss = conceptual_train_step(batch, multi_task_model, optimizer)
        epoch_total_loss += total_loss
        epoch_sft_loss += sft_loss
        epoch_reg_loss += reg_loss
        num_batches += 1

    avg_total_loss = epoch_total_loss / num_batches
    avg_sft_loss = epoch_sft_loss / num_batches
    avg_reg_loss = epoch_reg_loss / num_batches
    print(f"Epoch {epoch+1}/{num_epochs_example} | "
          f"Avg Total Loss: {avg_total_loss:.4f} | "
          f"Avg SFT Loss: {avg_sft_loss:.4f} | "
          f"Avg Regression Loss: {avg_reg_loss:.4f}")

print("\nConceptual training loop finished.")


# %% 6. Considerations and Freezing Layers (Conceptual)
print("\n--- 6. Considerations and Freezing Layers (Conceptual) ---\n")
print("1. Freezing Backbone Layers:")
print("   - If your SFT dataset is small or very different from pre-training, or if you want to")
print("     train only the new regression head initially, you can freeze the backbone.")
print("   - `for param in multi_task_model.backbone.parameters(): param.requires_grad = False`")
print("   - Then, only train the `regression_head` and potentially the `lm_head` if you're also adapting it.")
print("   - After some epochs, you might unfreeze some or all backbone layers for full fine-tuning.")

print("\n2. Differential Learning Rates:")
print("   - Use a smaller learning rate for the pre-trained backbone parameters and a larger one")
print("     for the newly initialized regression head parameters.")
# Example optimizer setup for differential LR:
# backbone_params = list(multi_task_model.backbone.parameters())
# regression_head_params = list(multi_task_model.regression_head.parameters())
# optimizer = optim.AdamW([
#     {'params': backbone_params, 'lr': 1e-5}, # Smaller LR for backbone
#     {'params': regression_head_params, 'lr': 1e-3} # Larger LR for new head
# ])

print("\n3. Pooling Strategy for Regression:")
print("   - The choice of pooling ('last', 'mean', 'max', or even attention-based pooling) for the")
print("     regression head can impact performance. 'mean' or 'last' (of actual content, not padding)")
print("     are common starting points.")

print("\n4. Task Weighting:")
print("   - The `sft_loss_weight` and `regression_loss_weight` are crucial hyperparameters.")
print("     They might need tuning. If one task's loss naturally has a much larger magnitude,")
print("     it might dominate; consider normalizing losses or adjusting weights accordingly.")

print("\n5. Evaluation:")
print("   - You'll need separate evaluation metrics for each task (e.g., perplexity or accuracy for SFT,")
print("     MSE or Pearson correlation for regression).")


print("\nEnd of Module 15.")