# 16_preference_optimization_dpo.py

# Module 16: Preference Optimization with DPO for LLMs
#
# This script explains Direct Preference Optimization (DPO) and demonstrates
# how to implement it to fine-tune a SmallGPT model using preference data.
# It builds upon the SmallGPT model defined in Module 08.
#
# We will cover:
# 1.  Conceptual Foundations of Preference Optimization:
#     - What is Preference Tuning?
#     - Direct Preference Optimization (DPO): Core Idea, Loss, Reference Model.
#     - Data format for DPO.
#     - Brief mention of GRPO and other techniques.
#     - Key implementation differences: DPO vs. SFT.
# 2.  Model Definition (SmallGPT).
# 3.  Checkpoint Loading.
# 4.  Data Preparation for DPO (Dataset, Collator).
# 5.  Setting up Models for DPO (Policy, Reference, Tokenizer, DataLoader).
# 6.  DPO Training Implementation (Log Probs, DPO Loss).
# 7.  Running a DPO Training Example.
# 8.  Inference Comparison: Original SFT vs. DPO-Tuned Model.
# 9.  Summary

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer # For tokenizer consistency
import random
import numpy as np
import os
import copy # For deep copying the reference model
from tqdm import tqdm
import time # For unique directory naming if saving DPO models

print("--- Module 16: Preference Optimization with DPO for LLMs ---\n")
print(f"Using PyTorch version: {torch.__version__}")

# --- Reproducibility & Device ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA (GPU) is available. Using device: {device}")
    torch.cuda.manual_seed_all(SEED)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"MPS is available. Using device: {device}")
else:
    device = torch.device("cpu")
    print(f"No GPU/MPS found. Using device: {device}")
print(f"Set random, numpy, and torch seeds to: {SEED}")


# --- Configuration ---
# These would typically be loaded from the SFT model's checkpoint,
# but we define them here for clarity if starting "fresh" conceptually.
# When loading a checkpoint, MODEL_CONFIG will be overridden.
MODEL_CONFIG = {
    'vocab_size': 50257,   # Placeholder, will be set by tokenizer or loaded checkpoint
    'embed_dim': 128,      # Must match the SFT model to be loaded
    'num_heads': 8,
    'num_layers': 6,
    'block_size': 128,     # Max sequence length (context window)
    'dropout_rate': 0.1
}

DPO_TRAIN_CONFIG = {
    'sft_model_checkpoint_path': "./checkpoints/run_20250515_105608_E1_S20000_D128H8L6/checkpoint_step_54000.pth", # IMPORTANT: SET THIS PATH TO YOUR TRAINED SmallGPT CHECKPOINT FROM MODULE 08
                                      # e.g., './checkpoints/run_YYYYMMDD_HHMMSS_E.../checkpoint_step_XXXX.pth'
    'batch_size': 2,        # Keep small for this demo due to synthetic data
    'learning_rate': 5e-6,  # DPO often uses smaller LRs than SFT
    'num_epochs': 3,        # Short for demo purposes
    'beta': 0.1,            # DPO loss temperature parameter
    'label_pad_token_id': -100, # Standard ignore_index for CrossEntropyLoss
    'max_seq_length': MODEL_CONFIG['block_size'], # Max length for prompt + completion
    'dpo_model_save_dir': './checkpoints_dpo', # Directory to save DPO model
}

# --- 1. Conceptual Foundations of Preference Optimization ---
print("\n--- 1. Conceptual Foundations of Preference Optimization ---\n")

print("1.1 What is Preference Tuning (Why beyond SFT)?")
print("Supervised Fine-Tuning (SFT) adapts LLMs to specific styles or tasks by training on high-quality demonstrations.")
print("However, SFT alone might not fully align the model with nuanced human preferences, especially regarding safety,")
print("helpfulness, or complex instructions. SFT trains the model to mimic outputs, not necessarily to understand 'good' vs 'bad'.")
print("\nPreference Tuning methods, like Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO),")
print("aim to steer LLM behavior more directly based on human judgments of response quality.")
print("Instead of just good examples, these methods use data indicating which of two (or more) responses is better for a given prompt.")

print("\n1.2 Direct Preference Optimization (DPO)")
print("   1.2.1 Core Idea and Benefits")
print("      DPO offers a more direct and often simpler way to optimize language models based on preference data.")
print("      Unlike traditional RLHF which involves training a separate reward model and then using RL (e.g., PPO)")
print("      to optimize the LLM against this reward model, DPO bypasses the explicit reward modeling step.")
print("      It directly optimizes the policy (the LLM being fine-tuned) to increase the likelihood of preferred")
print("      responses and decrease the likelihood of dispreferred ones, relative to a reference model.")
print("\n      Benefits of DPO:")
print("      - Simplicity: No need to train a separate reward model.")
print("      - Stability: Often more stable than complex RL training loops.")
print("      - Efficiency: Can be computationally less intensive than full RLHF.")

print("\n   1.2.2 The DPO Loss Function")
print("      The DPO loss is derived from a theoretical connection between reward functions and optimal policies.")
print("      It's formulated as a binary classification loss on preference pairs.")
print("      For a given prompt `x`, a chosen (preferred) completion `y_w`, and a rejected (dispreferred) completion `y_l`:")
print("      `L_DPO(π_θ, π_ref) = -E_((x, y_w, y_l)~D) [ log σ ( β * (log π_θ(y_w|x) - log π_ref(y_w|x)) - β * (log π_θ(y_l|x) - log π_ref(y_l|x)) ) ]`")
print("      Where:")
print("        - `π_θ` is the policy model (the LLM we are fine-tuning).")
print("        - `π_ref` is a reference model (typically the SFT model, kept frozen).")
print("        - `D` is the dataset of preference triplets `(x, y_w, y_l)`.")
print("        - `σ` is the sigmoid function.")
print("        - `β` (beta) is a temperature parameter that controls how much the policy model should deviate from the reference.")
print("          A higher `β` means stronger preference enforcement but potentially more deviation.")
print("      The term `log π(y|x)` represents the sum of log probabilities of the tokens in sequence `y` given prompt `x`.")
print("      The loss aims to make the term `(log π_θ(y_w|x) - log π_ref(y_w|x))` larger than `(log π_θ(y_l|x) - log π_ref(y_l|x))`.")
print("      Essentially, it encourages the policy model `π_θ` to assign a higher relative log-probability increase (compared to `π_ref`)")
print("      to chosen responses than to rejected responses.")

print("\n   1.2.3 The Reference Model (`π_ref`)")
print("      The reference model is crucial in DPO. It's typically a frozen copy of the model you start DPO training from (e.g., your SFT model).")
print("      Its role is to provide a stable baseline for log-probability comparisons.")
print("      This prevents the policy model `π_θ` from deviating too far from its initial (hopefully well-behaved) distribution,")
print("      which could lead to issues like mode collapse or generating gibberish while trying to satisfy preferences.")
print("      The reference model's outputs (log probabilities) are used in the loss calculation but its weights are NOT updated during DPO.")

print("\n1.3 Data for DPO: (Prompt, Chosen, Rejected) Triplets")
print("   DPO requires a dataset of preference triplets: `(prompt, chosen_response, rejected_response)`.")
print("   - `prompt`: The input text or instruction given to the LLM.")
print("   - `chosen_response`: The LLM-generated response that human evaluators preferred.")
print("   - `rejected_response`: The LLM-generated response that human evaluators dispreferred (or found less good).")
print("   Such datasets can be created by collecting human feedback on model generations or using existing structured datasets.")

print("\n1.4 Brief Mention of GRPO and Other Advanced Techniques")
print("   Direct Preference Optimization (DPO) is one of several methods for preference tuning.")
print("   - Generalized Preference Optimization (GRPO): An extension of DPO that aims to better handle scenarios")
print("     with ties or more complex preference structures (e.g., k-wise comparisons rather than just pairwise).")
print("     It might use a different loss formulation or allow for different divergence measures between policy and reference.")
print("     The Hugging Face TRL library provides an implementation of `GRPOTrainer`.")
print("   - Identity Preference Optimization (IPO): Another loss formulation that has shown strong performance,")
print("     aiming for a more direct regularization effect.")
print("   - Traditional RLHF (PPO): Involves training a reward model explicitly and then using RL algorithms like PPO.")
print("     While powerful, it can be more complex to implement and tune.")
print("\n   The field of LLM alignment is rapidly evolving, with new techniques and variations emerging regularly.")
print("   Libraries like Hugging Face TRL (`trl`) are excellent resources for implementations of these advanced methods.")

print("\n1.5 Key Implementation Differences: DPO vs. SFT")
print("   When implementing DPO, several aspects differ from a standard Supervised Fine-Tuning (SFT) setup:")
print("   1. Models: DPO requires two models: ")
print("      - Policy Model (`π_θ`): This is the model being actively trained. It's initialized from the SFT model.")
print("      - Reference Model (`π_ref`): A frozen copy of the initial SFT model. It's used to calculate baseline log-probabilities and is not updated.")
print("        In SFT, you typically only have one model being trained.")
print("   2. Data Format: DPO uses preference triplets `(prompt, chosen_response, rejected_response)`.")
print("      SFT typically uses `(prompt, ideal_completion)` pairs.")
print("   3. Log Probability Calculation: A core part of DPO involves calculating the log-probabilities of the chosen and rejected responses")
print("      under both the policy and reference models. This requires a specific function (like `get_sequence_log_probs` below).")
print("      In SFT, you directly compute loss (e.g., CrossEntropy) between model logits and target token IDs.")
print("   4. Loss Function: DPO uses its specific loss function (detailed in 1.2.2), which incorporates log-probabilities from both models and the `beta` parameter.")
print("      SFT typically uses Cross-Entropy Loss for next-token prediction.")
print("   5. Labeling: For DPO, labels for the `get_sequence_log_probs` function need to mask out the prompt tokens, so that log-probabilities are computed only for the response part.")
print("      While SFT also masks labels (e.g., for padding), the focus in DPO is specifically on the response segments of chosen/rejected sequences.")


# --- 2. Model Definition (SmallGPT from Module 08) ---
print("\n--- 2. Model Definition (SmallGPT from Module 08) ---\n")

class TransformerBlock(nn.Module):
    """A single Transformer block for a decoder-style GPT."""
    def __init__(self, embed_dim, num_heads, ff_dim_multiplier=4, dropout_rate=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        ff_dim = embed_dim * ff_dim_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout_rate)
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        _batch_size, seq_len, _embed_dim = x.shape
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            sz=seq_len, device=x.device # Removed dtype=x.dtype to use default bool, MHA handles it.
        )
        norm_x = self.ln1(x)
        attn_output, _ = self.attn(norm_x, norm_x, norm_x,
                                   attn_mask=causal_mask,
                                   is_causal=False, # We provide the causal mask explicitly
                                   need_weights=False)
        x = x + self.dropout(attn_output)
        norm_x = self.ln2(x)
        ff_output = self.ffn(norm_x)
        x = x + ff_output # FFN's residual includes dropout from its definition
        return x

class SmallGPT(nn.Module):
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
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.token_embedding.weight = self.lm_head.weight # Tie weights
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                if name == 'lm_head.weight' or name == 'token_embedding.weight' or name == 'positional_embedding.weight':
                     nn.init.normal_(param, mean=0.0, std=0.02)
                else: # Other weights like in Linear layers of FFN or MHA
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, idx_sequences, attention_mask=None): # Added attention_mask compatibility (though not used by current MHA)
        # `attention_mask` is not strictly used by this SmallGPT version's MHA if all inputs are dense and up to block_size.
        # However, having it as an argument is good practice for future extensions or if using key_padding_mask.
        # For DPO, we will pad sequences to `block_size`, so a full attention_mask isn't critical for this specific MHA.
        # The causal mask is internally generated.
        batch_size, seq_len = idx_sequences.shape
        tok_emb = self.token_embedding(idx_sequences)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=idx_sequences.device)
        pos_emb = self.positional_embedding(positions[:seq_len]) # Ensure positions don't exceed block_size if seq_len < block_size for some reason
        x = tok_emb + pos_emb
        x = self.dropout(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits

print("SmallGPT and TransformerBlock classes defined.")

# --- 3. Checkpoint Loading Helper (adapted from Module 08) ---
print("\n--- 3. Checkpoint Loading Helper ---\n")

def load_sft_checkpoint(model_class, checkpoint_path_to_load, device_to_load_on):
    global MODEL_CONFIG # Allow modification of global MODEL_CONFIG
    if checkpoint_path_to_load is None or not os.path.exists(checkpoint_path_to_load):
        if checkpoint_path_to_load:
            print(f"SFT Checkpoint path '{checkpoint_path_to_load}' not found.")
        else:
            print("No SFT checkpoint path specified.")
        raise FileNotFoundError("SFT Checkpoint is required for DPO. Please set DPO_TRAIN_CONFIG['sft_model_checkpoint_path'].")

    print(f"Loading SFT checkpoint from {checkpoint_path_to_load}...")
    checkpoint = torch.load(checkpoint_path_to_load, map_location=device_to_load_on)

    loaded_model_config = checkpoint['model_config']
    MODEL_CONFIG = loaded_model_config # Update global MODEL_CONFIG
    DPO_TRAIN_CONFIG['max_seq_length'] = MODEL_CONFIG['block_size'] # Sync max_seq_length

    model = model_class(
        vocab_size=MODEL_CONFIG['vocab_size'],
        embed_dim=MODEL_CONFIG['embed_dim'],
        num_heads=MODEL_CONFIG['num_heads'],
        num_layers=MODEL_CONFIG['num_layers'],
        block_size=MODEL_CONFIG['block_size'],
        dropout_rate=MODEL_CONFIG['dropout_rate']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device_to_load_on)

    epoch = checkpoint.get('epoch', -1)
    global_step = checkpoint.get('global_step', 0)
    loss = checkpoint.get('loss', float('inf'))

    print(f"SFT Checkpoint loaded. Model was trained for {epoch+1} epochs, {global_step} steps. Last SFT loss: {loss:.4f}")
    print(f"Using Model Config from SFT checkpoint: {MODEL_CONFIG}")
    return model

print("Checkpoint loading helper defined.")


# --- 4. Data Preparation for DPO ---
print("\n--- 4. Data Preparation for DPO ---\n")

print("4.1 Dummy Preference Dataset Implementation")
dummy_preference_data = [
    {
        "prompt": "Explain the concept of photosynthesis in simple terms.",
        "chosen": "Photosynthesis is how plants make their own food using sunlight, water, and air. It's like they're tiny chefs!",
        "rejected": "Photosynthesis is a complex biochemical process involving light-dependent and light-independent reactions converting light energy into chemical energy stored in glucose."
    },
    {
        "prompt": "What is the capital of France?",
        "chosen": "The capital of France is Paris.",
        "rejected": "France's capital city is London. Just kidding, it's Paris!" # Slightly worse due to joke
    },
    {
        "prompt": "Write a short poem about a cat.",
        "chosen": "Golden eyes, a gentle purr, \nSoft fur warmed by morning's blur. \nA hunter sly, a friend so true, \nThat's my cat, I love you.",
        "rejected": "Cats are animals. They meow. They sleep a lot. Some are fluffy."
    },
    # Add more diverse examples if desired for a more robust dummy run
    {
        "prompt": "Translate 'hello' to Spanish.",
        "chosen": "Hola.",
        "rejected": "Bonjourno." # Incorrect
    }
]

class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_length, label_pad_token_id):
        self.prompts_text = [item["prompt"] for item in data]
        self.chosen_text = [item["chosen"] for item in data]
        self.rejected_text = [item["rejected"] for item in data]
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.label_pad_token_id = label_pad_token_id
        print(f"PreferenceDataset initialized with {len(data)} samples.")

    def __len__(self):
        return len(self.prompts_text)

    def _prepare_sequence(self, prompt_text, response_text):
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)

        # Concatenate prompt and response, add EOS if tokenizer has one
        # (GPT2 tokenizer usually handles EOS via encode if it's part of its training)
        # For explicit control, one might add tokenizer.eos_token_id if not present in response_ids[-1]
        full_ids = prompt_ids + response_ids
        if self.tokenizer.eos_token_id is not None and (not response_ids or response_ids[-1] != self.tokenizer.eos_token_id):
             full_ids += [self.tokenizer.eos_token_id]


        # Truncate if too long (from the end of prompt, keeping response intact if possible, or truncating response)
        if len(full_ids) > self.max_seq_length:
            # Simple truncation: from the end. More sophisticated strategies could be used.
            # Here, we prioritize keeping the response, so truncate from start of prompt if prompt+response is too long.
            # A common strategy is to truncate prompt if prompt + response > max_length.
            excess_length = len(full_ids) - self.max_seq_length
            if excess_length > 0:
                full_ids = full_ids[excess_length:] # Truncate from the beginning
                # Update prompt_len after truncation if needed, but for labels we only need original prompt_len
                # This simple truncation might remove all prompt if response is too long.
                # A better way for very long responses: truncate response, or truncate prompt keeping some.
                # For this demo, let's assume max_seq_length is reasonably large for dummy data.
                # A safer truncation:
                # prompt_part = full_ids[:len(prompt_ids)]
                # response_part = full_ids[len(prompt_ids):]
                # if len(prompt_part) + len(response_part) > self.max_seq_length:
                #    available_for_prompt = self.max_seq_length - len(response_part)
                #    if available_for_prompt < 0: # response itself is too long
                #        response_part = response_part[:self.max_seq_length]
                #        prompt_part = []
                #    else:
                #        prompt_part = prompt_part[-available_for_prompt:] # keep end of prompt
                # full_ids = prompt_part + response_part

        input_ids = full_ids
        attention_mask = [1] * len(input_ids)

        # Create labels: prompt tokens are masked with label_pad_token_id
        # Labels are shifted by one position relative to input_ids for next-token prediction
        # So, labels for input_ids[i] is input_ids[i+1]
        # Prompt part of labels should be label_pad_token_id
        labels = [self.label_pad_token_id] * len(prompt_ids) + response_ids[:] # Use original response_ids for labels
        if self.tokenizer.eos_token_id is not None and (not response_ids or response_ids[-1] != self.tokenizer.eos_token_id):
            labels += [self.tokenizer.eos_token_id]

        # Truncate labels similar to input_ids if truncation happened
        if len(input_ids) < len(labels): # Should not happen with current full_ids truncation
            labels = labels[:len(input_ids)]
        elif len(labels) < len(input_ids) and len(labels) == len(prompt_ids) + len(response_ids): # EOS was added to input_ids but not labels
             pass # This case should be handled by consistent EOS addition

        # Ensure labels are consistent with truncated input_ids. If full_ids was truncated from start.
        if len(full_ids) < len(prompt_ids) + len(response_ids) + (1 if self.tokenizer.eos_token_id else 0): # truncation happened
            # This part is tricky if prompt is truncated.
            # For simplicity, let's re-evaluate prompt_len based on truncated input_ids,
            # assuming response is always at the end.
            # This is NOT robust. A better way is to tokenize prompt and response separately and then combine & truncate.
            # The current TRL library approach: tokenize prompt, tokenize response. Truncate prompt if prompt+response > max_len.
            # Then concatenate. This is more robust.
            # For this demo, the dummy data is short, so truncation is less likely an issue.
            # Let's assume len(prompt_ids) is for the *original untruncated* prompt for label masking.
            # The `get_sequence_log_probs` will handle masking based on `labels`.
            pass

        return input_ids, attention_mask, labels, len(prompt_ids)

    def __getitem__(self, idx):
        prompt_text = self.prompts_text[idx]
        chosen_text = self.chosen_text[idx]
        rejected_text = self.rejected_text[idx]

        chosen_input_ids, chosen_attention_mask, chosen_labels, chosen_prompt_len = \
            self._prepare_sequence(prompt_text, chosen_text)
        
        rejected_input_ids, rejected_attention_mask, rejected_labels, rejected_prompt_len = \
            self._prepare_sequence(prompt_text, rejected_text)

        # Sanity check prompt lengths (should be same for chosen/rejected from same prompt)
        assert chosen_prompt_len == rejected_prompt_len, "Prompt length mismatch between chosen and rejected processing."

        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "chosen_labels": chosen_labels, # These labels have prompt part masked
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
            "rejected_labels": rejected_labels, # These labels have prompt part masked
            "prompt_len": chosen_prompt_len # Original length of prompt tokens
        }


print("\n5.3 Data Collator for DPO")
class PreferenceDataCollator:
    def __init__(self, tokenizer, max_seq_length, label_pad_token_id):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.label_pad_token_id = label_pad_token_id
        if self.tokenizer.pad_token_id is None:
            print("Warning: tokenizer.pad_token_id is None. Setting to eos_token_id for padding.")
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        assert self.tokenizer.pad_token_id is not None, "Tokenizer must have a pad_token_id"

    def _pad_and_truncate(self, sequences, pad_value, max_len):
        padded_sequences = []
        for seq in sequences:
            if len(seq) > max_len:
                # Truncate from the end. For DPO, we prefer truncating completions if prompt+completion is too long.
                # However, _prepare_sequence already did some truncation. This is a final safety net.
                # A common strategy is to truncate from the start of the sequence if it's input_ids,
                # or from the end of the prompt part.
                # Here, simple truncation from the end for demo.
                seq = seq[:max_len]
            
            padding_len = max_len - len(seq)
            padded_sequences.append(seq + [pad_value] * padding_len)
        return padded_sequences

    def __call__(self, batch):
        collated_batch = {}

        # Handle chosen sequences
        chosen_input_ids = [item["chosen_input_ids"] for item in batch]
        chosen_labels = [item["chosen_labels"] for item in batch] # Labels already have prompt masked
        chosen_attention_masks = [[1] * len(ids) for ids in chosen_input_ids] # Dynamic mask before padding

        collated_batch["chosen_input_ids"] = torch.tensor(
            self._pad_and_truncate(chosen_input_ids, self.tokenizer.pad_token_id, self.max_seq_length),
            dtype=torch.long
        )
        collated_batch["chosen_attention_mask"] = torch.tensor(
            self._pad_and_truncate(chosen_attention_masks, 0, self.max_seq_length), # Pad attention mask with 0
            dtype=torch.long
        )
        # For labels, pad with label_pad_token_id, as these parts should be ignored in loss
        collated_batch["chosen_labels"] = torch.tensor(
            self._pad_and_truncate(chosen_labels, self.label_pad_token_id, self.max_seq_length),
            dtype=torch.long
        )
        
        # Handle rejected sequences
        rejected_input_ids = [item["rejected_input_ids"] for item in batch]
        rejected_labels = [item["rejected_labels"] for item in batch] # Labels already have prompt masked
        rejected_attention_masks = [[1] * len(ids) for ids in rejected_input_ids]

        collated_batch["rejected_input_ids"] = torch.tensor(
            self._pad_and_truncate(rejected_input_ids, self.tokenizer.pad_token_id, self.max_seq_length),
            dtype=torch.long
        )
        collated_batch["rejected_attention_mask"] = torch.tensor(
            self._pad_and_truncate(rejected_attention_masks, 0, self.max_seq_length),
            dtype=torch.long
        )
        collated_batch["rejected_labels"] = torch.tensor(
            self._pad_and_truncate(rejected_labels, self.label_pad_token_id, self.max_seq_length),
            dtype=torch.long
        )
        
        # Note: "prompt_len" from dataset items is not directly used here as labels already encode this.
        # If needed for other purposes, it could be collated too.
        return collated_batch

print("PreferenceDataset and PreferenceDataCollator defined.")

# --- 5. Setting up Models for DPO ---
print("\n--- 5. Setting up Models for DPO ---\n")

print("5.1 Load Tokenizer")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set tokenizer.pad_token to tokenizer.eos_token ({tokenizer.eos_token})")
MODEL_CONFIG['vocab_size'] = tokenizer.vocab_size # Set initial vocab size based on tokenizer

print("\n5.2 Load Pre-trained SFT Model (Policy Model)")
if DPO_TRAIN_CONFIG['sft_model_checkpoint_path'] is None:
    print("ERROR: DPO_TRAIN_CONFIG['sft_model_checkpoint_path'] is not set.")
    print("Please set this to the path of a .pth checkpoint file from Module 08 training.")
    print("Skipping DPO setup and training.")
    # Exit or skip further execution if no SFT model path is provided
    policy_model = None 
else:
    try:
        policy_model = load_sft_checkpoint(SmallGPT, DPO_TRAIN_CONFIG['sft_model_checkpoint_path'], device)
        policy_model.train() # Set to train mode
        print("SFT model loaded successfully as policy_model.")
    except FileNotFoundError as e:
        print(f"Error loading SFT model: {e}")
        policy_model = None
    except Exception as e:
        print(f"An unexpected error occurred while loading the SFT model: {e}")
        policy_model = None

print("\n5.3 Create Reference Model (Frozen copy of SFT model)")
if policy_model:
    reference_model = copy.deepcopy(policy_model)
    reference_model.eval() # Set to eval mode
    for param in reference_model.parameters():
        param.requires_grad = False
    print("Reference model created as a frozen copy of the policy model.")
else:
    reference_model = None
    print("Skipping reference model creation as policy model failed to load.")

print("\n5.4 Instantiate Preference Dataset and DataLoader")
if policy_model: 
    DPO_TRAIN_CONFIG['max_seq_length'] = MODEL_CONFIG['block_size']

    # Re-initialize dataset & collator with potentially updated max_seq_length from loaded model
    preference_dataset = PreferenceDataset(
        dummy_preference_data,
        tokenizer,
        DPO_TRAIN_CONFIG['max_seq_length'],
        DPO_TRAIN_CONFIG['label_pad_token_id']
    )
    data_collator = PreferenceDataCollator(
        tokenizer,
        DPO_TRAIN_CONFIG['max_seq_length'],
        DPO_TRAIN_CONFIG['label_pad_token_id']
    )
    preference_dataloader = DataLoader(
        preference_dataset,
        batch_size=DPO_TRAIN_CONFIG['batch_size'],
        collate_fn=data_collator,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED) # For reproducible shuffling
    )
    print(f"Preference DataLoader created with batch size {DPO_TRAIN_CONFIG['batch_size']}.")
    
    # Test one batch from dataloader
    try:
        test_batch = next(iter(preference_dataloader))
        print("Sample batch keys:", test_batch.keys())
        print("Sample chosen_input_ids shape:", test_batch["chosen_input_ids"].shape)
        print("Sample chosen_labels shape:", test_batch["chosen_labels"].shape)
    except Exception as e:
        print(f"Error when testing dataloader: {e}")
        print("Ensure sft_model_checkpoint_path is correctly set and model loaded.")
else:
    preference_dataloader = None
    print("Skipping dataset/dataloader creation as models are not available.")

# --- 6. DPO Training Implementation ---
print("\n--- 6. DPO Training Implementation ---\n")

print("6.1 Function to get Log Probabilities of Sequences")
def get_sequence_log_probs(model, input_ids, labels, attention_mask, label_pad_token_id):
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    attention_mask = attention_mask.to(device)

    # Get logits from the model
    # Logits shape: (batch_size, seq_len, vocab_size)
    all_logits = model(input_ids, attention_mask=attention_mask)

    # Shift logits and labels for next token prediction
    # Logits for token i predict token i+1.
    # So, logits[:, :-1, :] are for predicting labels[:, 1:]
    shifted_logits = all_logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    
    # Calculate log probabilities of actual target tokens
    log_probs = F.log_softmax(shifted_logits, dim=-1)

    # Gather the log_probs of the true tokens from labels
    # Need to handle label_pad_token_id (-100) which might be out of vocab range for gather
    # Clamp shifted_labels to be valid indices for gather, then use mask for actual sum
    valid_labels_for_gather = shifted_labels.clone()
    valid_labels_for_gather[valid_labels_for_gather == label_pad_token_id] = 0 # Replace -100 with a valid index (e.g., 0)
    
    token_log_probs = torch.gather(log_probs, -1, valid_labels_for_gather.unsqueeze(-1)).squeeze(-1)

    # Create a mask for actual completion tokens (where labels are not pad_token_id)
    completion_mask = (shifted_labels != label_pad_token_id).float()

    # Sum log probabilities for completion tokens only
    # sequence_log_probs will have shape (batch_size,)
    sequence_log_probs = (token_log_probs * completion_mask).sum(dim=-1)
    
    return sequence_log_probs

print("get_sequence_log_probs function defined.")

print("\n6.2 DPO Loss Calculation")
def calculate_dpo_loss(policy_chosen_logps, policy_rejected_logps,
                       ref_chosen_logps, ref_rejected_logps, beta):
    """
    Calculates the DPO loss.
    All logps are 1D tensors of shape (batch_size,).
    """
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    
    # The DPO loss is -log_sigmoid(chosen_rewards - rejected_rewards)
    # This is equivalent to sigmoid_cross_entropy with logits=(chosen_rewards - rejected_rewards) and labels=1
    # Or: loss = -F.logsigmoid(chosen_rewards - rejected_rewards)
    # For numerical stability with F.binary_cross_entropy_with_logits:
    # logits = chosen_rewards - rejected_rewards
    # loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction='none')
    # However, -F.logsigmoid is standard.
    
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards)
    return loss.mean() # Average over the batch

print("calculate_dpo_loss function defined.")

print("\n6.3 The DPO Training Loop (to be run in next section)")

# --- 7. Running a DPO Example ---
print("\n--- 7. Running a DPO Example ---\n")

if policy_model and reference_model and preference_dataloader:
    print("Starting DPO training example...")
    optimizer = optim.AdamW(policy_model.parameters(), lr=DPO_TRAIN_CONFIG['learning_rate'])
    
    policy_model.train() # Ensure policy model is in training mode
    reference_model.eval() # Ensure reference model is in eval mode

    total_steps = 0
    avg_epoch_loss = 0 # Initialize in case of no epochs
    for epoch in range(DPO_TRAIN_CONFIG['num_epochs']):
        print(f"\nEpoch {epoch+1}/{DPO_TRAIN_CONFIG['num_epochs']}")
        epoch_loss_sum = 0
        num_batches_in_epoch = 0

        progress_bar = tqdm(preference_dataloader, desc=f"Epoch {epoch+1}", unit="batch")

        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()

            # Get log probabilities from policy model
            policy_chosen_logps = get_sequence_log_probs(
                policy_model,
                batch["chosen_input_ids"],
                batch["chosen_labels"],
                batch["chosen_attention_mask"],
                DPO_TRAIN_CONFIG['label_pad_token_id']
            )
            policy_rejected_logps = get_sequence_log_probs(
                policy_model,
                batch["rejected_input_ids"],
                batch["rejected_labels"],
                batch["rejected_attention_mask"],
                DPO_TRAIN_CONFIG['label_pad_token_id']
            )

            # Get log probabilities from reference model (with no_grad)
            with torch.no_grad():
                ref_chosen_logps = get_sequence_log_probs(
                    reference_model,
                    batch["chosen_input_ids"],
                    batch["chosen_labels"],
                    batch["chosen_attention_mask"],
                    DPO_TRAIN_CONFIG['label_pad_token_id']
                )
                ref_rejected_logps = get_sequence_log_probs(
                    reference_model,
                    batch["rejected_input_ids"],
                    batch["rejected_labels"],
                    batch["rejected_attention_mask"],
                    DPO_TRAIN_CONFIG['label_pad_token_id']
                )
            
            # Calculate DPO loss
            loss = calculate_dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps,
                DPO_TRAIN_CONFIG['beta']
            )

            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item()
            num_batches_in_epoch +=1
            total_steps += 1
            progress_bar.set_postfix(loss=loss.item(), total_steps=total_steps)

        avg_epoch_loss = epoch_loss_sum / num_batches_in_epoch if num_batches_in_epoch > 0 else 0
        print(f"End of Epoch {epoch+1}, Average DPO Loss: {avg_epoch_loss:.4f}")

    print("\nDPO training example finished.")

    # --- Optional: Save the DPO-tuned model ---
    dpo_model_save_dir = DPO_TRAIN_CONFIG['dpo_model_save_dir']
    if not os.path.exists(dpo_model_save_dir):
        os.makedirs(dpo_model_save_dir)
    
    dpo_model_filename = f"smallgpt_dpo_tuned_E{DPO_TRAIN_CONFIG['num_epochs']}_B{DPO_TRAIN_CONFIG['beta']}_{time.strftime('%Y%m%d_%H%M%S')}.pth"
    dpo_model_save_path = os.path.join(dpo_model_save_dir, dpo_model_filename)
    
    torch.save({
        'model_state_dict': policy_model.state_dict(),
        'model_config': MODEL_CONFIG, # Save the config used for this model
        'dpo_train_config': DPO_TRAIN_CONFIG,
        'final_loss': avg_epoch_loss,
        'total_steps': total_steps
    }, dpo_model_save_path)
    print(f"DPO-tuned policy model saved to: {dpo_model_save_path}")

else:
    print("Skipping DPO training example because SFT model (policy_model) or reference_model could not be loaded/created.")
    print("Please ensure DPO_TRAIN_CONFIG['sft_model_checkpoint_path'] points to a valid SmallGPT checkpoint from Module 08.")

# --- 8. Inference Comparison: Original SFT vs. DPO-Tuned Model ---
print("\n--- 8. Inference Comparison: Original SFT vs. DPO-Tuned Model ---\n")

def generate_text_dpo(model_instance, start_string, tokenizer_instance, max_len=60, temperature=0.7, top_k=40):
    """
    Generates text from a model.
    MODEL_CONFIG and device are accessed globally.
    """
    
    model_instance.eval() # Ensure model is in eval mode
    
    input_ids = tokenizer_instance.encode(start_string, add_special_tokens=False)
    generated_ids = list(input_ids) 

    # Context for the model should be (batch_size, seq_len)
    # Ensure input_ids are within block_size for model context
    current_context_ids_list = input_ids[-MODEL_CONFIG['block_size']:]
    current_context_ids = torch.tensor([current_context_ids_list], dtype=torch.long, device=device)

    with torch.no_grad():
        for i in range(max_len):
            logits = model_instance(current_context_ids) # (1, current_seq_len, vocab_size)
            
            next_token_logits = logits[:, -1, :] # Focus on the logits for the last token (1, vocab_size)

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            
            if temperature > 0: # For temperature=0, one might use argmax before softmax for pure greedy.
                next_token_logits = next_token_logits / temperature
            
            probs = F.softmax(next_token_logits, dim=-1) # (1, vocab_size)
            
            # Sample the next token ID
            next_token_id = torch.multinomial(probs, num_samples=1).item() # Get scalar
            
            generated_ids.append(next_token_id)
            
            # Update context: append new token, and trim to block_size if needed
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            current_context_ids = torch.cat((current_context_ids, next_token_tensor), dim=1)
            if current_context_ids.shape[1] > MODEL_CONFIG['block_size']:
                current_context_ids = current_context_ids[:, -MODEL_CONFIG['block_size']:]
            
            if tokenizer_instance.eos_token_id is not None and next_token_id == tokenizer_instance.eos_token_id:
                # print(f"EOS token ({tokenizer_instance.eos_token_id}) generated at step {i+1}. Stopping.") # Can be verbose
                break
            
    return tokenizer_instance.decode(generated_ids)

if policy_model and reference_model and tokenizer:
    print("Running inference tests...")
    
    inference_prompts = [
        "Explain the concept of photosynthesis in simple terms.",
        "What is the capital of France? Tell me something interesting about it.",
        "Write a short poem about a dog."
    ]

    for i, prompt_text in enumerate(inference_prompts):
        print(f"\n--- Inference Test {i+1} ---")
        print(f"Prompt: {prompt_text}")

        print("\nOriginal SFT Model (Reference Model) Output:")
        try:
            reference_model.to(device) # Ensure model is on the correct device
            original_output = generate_text_dpo(reference_model, prompt_text, tokenizer)
            print(original_output)
        except Exception as e:
            print(f"Error during generation with reference_model: {e}")

        print("\nDPO-Tuned Model (Policy Model) Output:")
        try:
            policy_model.to(device) # Ensure model is on the correct device
            dpo_output = generate_text_dpo(policy_model, prompt_text, tokenizer)
            print(dpo_output)
        except Exception as e:
            print(f"Error during generation with policy_model: {e}")
        print("-----------------------------")
else:
    print("Skipping inference comparison as models or tokenizer are not available.")

# --- 9. Summary ---
print("\n--- 9. Summary ---\n")
print("This module introduced Direct Preference Optimization (DPO) as a method for fine-tuning LLMs based on human preferences.")
print("Key takeaways:")
print("  - DPO directly optimizes a policy model using preference pairs (chosen, rejected) without needing an explicit reward model.")
print("  - It relies on a frozen reference model (usually the SFT model) to regularize training.")
print("  - The DPO loss function encourages the policy to assign higher relative log-probabilities to chosen responses than rejected ones.")
print("  - We demonstrated how to set up the data, models (policy and reference), and a basic DPO training loop using PyTorch")
print("    with the `SmallGPT` architecture from Module 08.")
print("  - Inference comparison can help illustrate behavioral changes post-DPO.")
print("\nPreference tuning is a critical step for creating LLMs that are not only capable but also aligned with desired behaviors")
print("like helpfulness, harmlessness, and adherence to instructions.")

print("\nEnd of Module 16.")
