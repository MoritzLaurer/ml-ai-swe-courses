# -*- coding: utf-8 -*-
# 12_fine_tuning_pretrained_models.py

# Module 12: Fine-tuning Pre-trained Models
#
# This script focuses on the concepts and strategies involved in fine-tuning
# pre-trained language models for downstream tasks.
#
# We will cover:
# 1. What is Fine-tuning? The "Why" and "When".
# 2. Core Strategy: Leveraging a Pre-trained Backbone.
# 3. Modifying the Model for a New Task (e.g., adding a classification head).
# 4. Freezing and Unfreezing Layers: Feature Extraction vs. Full Fine-tuning.
# 5. Differential Learning Rates for different model parts.
# 6. Data Preparation for the Downstream Task.
# 7. The Fine-tuning Loop.
# 8. Conceptual Introduction to Parameter-Efficient Fine-Tuning (PEFT) and LoRA.
# 9. Connecting these concepts to the Hugging Face `transformers` library.
# 10. Brief JAX Comparison.
#
# NOTE: This module is primarily conceptual and illustrates how you would adapt
# a pre-trained model. Loading specific checkpoints from previous modules can be
# done, but the focus is on the fine-tuning process itself.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel # For Hugging Face examples
import os

print("--- Module 12: Fine-tuning Pre-trained Models ---\n")
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

# %% 1. What is Fine-tuning? The "Why" and "When".
print("\n--- 1. What is Fine-tuning? The \"Why\" and \"When\". ---\n")

print("Fine-tuning is the process of taking a model that has already been trained on a large dataset")
print("(a 'pre-trained' model) and then training it further on a smaller, task-specific dataset.")

print("\nWhy fine-tune?")
print("- Leverage Pre-trained Knowledge: Models like GPT, BERT, LLaMA are trained on vast amounts of text data,")
print("  learning general language understanding, grammar, facts, and reasoning abilities.")
print("  Fine-tuning allows you to transfer this knowledge to your specific task without")
print("  training a large model from scratch, which is computationally expensive and data-hungry.")
print("- Achieve Better Performance: For many NLP tasks, fine-tuning a pre-trained model often yields")
print("  state-of-the-art results, especially when task-specific data is limited.")
print("- Faster Convergence: Since the model already has a good understanding of language,")
print("  it typically converges faster on the downstream task compared to training from random initialization.")

print("\nWhen to fine-tune?")
print("- When you have a specific downstream task (e.g., sentiment analysis, question answering, text summarization,")
print("  code generation for a specific style) and a relevant dataset for it.")
print("- When training a model from scratch is infeasible due to limited data or computational resources.")
print("- When you want to adapt a general-purpose LLM to a particular domain, style, or task.")


# %% 2. Core Strategy: Leveraging a Pre-trained Backbone
print("\n--- 2. Core Strategy: Leveraging a Pre-trained Backbone ---\n")

print("The core idea is to use the majority of the pre-trained model (its 'backbone') as a feature extractor.")
print("The backbone typically consists of the embedding layers and the main transformer blocks.")
print("These layers have learned rich representations of language.")

# --- Conceptual Model Loading ---
# Imagine we have the SmallGPT model class from Module 8.
# We would load its pre-trained weights.
# (Loading mechanics were covered in Module 8 and 9; here we assume it's done)

# Example: Placeholder for model definition (from Module 8)
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim_multiplier=4, dropout_rate=0.1): # Simplified
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        ff_dim = embed_dim * ff_dim_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), 
            nn.ReLU(), 
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout_rate) # Dropout is part of FFN
        )
        self.dropout = nn.Dropout(dropout_rate) # For attention output

    # Placeholder for the full forward method - actual implementation would be more complex
    def forward(self, x):
        # Simplified conceptual forward pass for placeholder
        # Actual implementation needs causal mask etc. as in Module 8
        _batch_size, seq_len, _embed_dim = x.shape
        
        # Conceptual Self-Attention part
        norm_x_attn = self.ln1(x)
        # In a real scenario, causal_mask would be generated and passed here
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            sz=seq_len, device=x.device, dtype=x.dtype
        )
        attn_output, _ = self.attn(norm_x_attn, norm_x_attn, norm_x_attn, 
                                   attn_mask=causal_mask, is_causal=False, need_weights=False)
        x = x + self.dropout(attn_output) # Dropout after attention

        # Conceptual FFN part
        norm_x_ffn = self.ln2(x)
        ff_output = self.ffn(norm_x_ffn) # ff_output already has dropout from self.ffn
        x = x + ff_output # No redundant self.dropout() here
        return x


class SmallGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, block_size, dropout_rate, num_classes=None):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(block_size, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, dropout_rate=dropout_rate) for _ in range(num_layers)]
        )
        self.ln_final = nn.LayerNorm(embed_dim)
        
        # Original LM head (predicts next token)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        # Tie weights
        self.token_embedding.weight = self.lm_head.weight
        
        # New: Optional classification head for fine-tuning
        self.num_classes = num_classes
        if self.num_classes is not None:
            self.classification_head = nn.Linear(embed_dim, num_classes)

    def forward(self, idx_sequences, task_type="language_modeling"):
        # ... (embedding and transformer blocks logic as in Module 8) ...
        batch_size, seq_len = idx_sequences.shape
        tok_emb = self.token_embedding(idx_sequences)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=idx_sequences.device)
        pos_emb = self.positional_embedding(positions)
        x = tok_emb + pos_emb
        x = self.dropout(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.ln_final(x) # Shape: (batch, seq_len, embed_dim)

        if task_type == "language_modeling":
            return self.lm_head(x)
        elif task_type == "classification" and self.num_classes is not None:
            # For classification, often use the representation of a specific token (e.g., [CLS])
            # or an aggregation (e.g., mean pooling) of the last hidden states.
            # Here, let's use the last token's representation from the sequence for simplicity.
            pooled_output = x[:, -1, :] # (batch, embed_dim)
            return self.classification_head(pooled_output) # (batch, num_classes)
        else:
            raise ValueError(f"Unsupported task_type: {task_type} or num_classes not set for classification.")

print("Pre-trained model structure (e.g., SmallGPT) would be loaded.")
# Assume `pretrained_model_config` and `pretrained_state_dict` are loaded.
# model = SmallGPT(**pretrained_model_config)
# model.load_state_dict(pretrained_state_dict)


# %% 3. Modifying the Model for a New Task
print("\n--- 3. Modifying the Model for a New Task ---\n")

print("For many downstream tasks, the original output layer of the pre-trained model")
print("(e.g., the language modeling head predicting the next token) is not suitable.")
print("You typically replace or add a new 'head' layer specific to your task.")

print("\nExample: Fine-tuning for Text Classification (e.g., Sentiment Analysis)")
print("- Task: Classify a piece of text into N categories (e.g., positive, negative, neutral).")
print("- Modification: Remove the pre-trained LM head. Add a new linear layer (classification head)")
print("  that takes the model's hidden state representation and outputs logits for N classes.")
print("  Input to this head is often the hidden state of a special [CLS] token, or an aggregation")
print("  (like mean/max pooling) of all token representations from the last layer.")

# --- Modifying our SmallGPT example ---
# model_config_from_pretrained = {'vocab_size': 50257, 'embed_dim': 128, 'num_heads': 8, 'num_layers': 6, 'block_size': 128, 'dropout_rate': 0.1} # Example
# num_sentiment_classes = 3 # e.g., positive, negative, neutral

# Load the pre-trained model (conceptually)
# pretrained_gpt = SmallGPT(**model_config_from_pretrained, num_classes=None)
# Load state_dict for pretrained_gpt here...
# pretrained_gpt.to(device)

# Now, adapt it for classification
# Option 1: If SmallGPT class supports `num_classes` argument (as modified above)
# classification_model = SmallGPT(**model_config_from_pretrained, num_classes=num_sentiment_classes)
# # Copy weights from pretrained backbone, excluding the original lm_head and the new classification_head (which is randomly initialized)
# backbone_state_dict = {k: v for k, v in pretrained_gpt.state_dict().items() if not k.startswith('lm_head')}
# classification_model.load_state_dict(backbone_state_dict, strict=False) # strict=False allows missing/extra keys
# classification_model.to(device)
# print(f"\nConceptual: Model modified with a new classification head for {num_sentiment_classes} classes.")

print("\nAnother common approach is to take the pre-trained backbone and add a new module:")
# class SentimentClassifier(nn.Module):
#     def __init__(self, backbone_model, hidden_dim, num_classes):
#         super().__init__()
#         self.backbone = backbone_model
#         # Freeze backbone if desired (see next section)
#         self.classifier_head = nn.Linear(hidden_dim, num_classes) # hidden_dim is backbone's embed_dim
#
#     def forward(self, idx_sequences):
#         # Get backbone embeddings (ignoring its original lm_head)
#         # This requires backbone to output hidden states before its own head.
#         # For SmallGPT, we'd need to ensure it can return 'x' before lm_head.
#         hidden_states = self.backbone(idx_sequences, task_type="get_hidden_states_only") # Hypothetical modification
#         pooled_output = hidden_states[:, -1, :] # Use last token's output
#         logits = self.classifier_head(pooled_output)
#         return logits


# %% 4. Freezing and Unfreezing Layers
print("\n--- 4. Freezing and Unfreezing Layers ---\n")

print("When fine-tuning, you can choose which parts of the model are updated.")
print("- Freezing Layers: Setting `param.requires_grad = False` for parameters in certain layers.")
print("  Frozen layers do not update their weights during training. They act as fixed feature extractors.")
print("- Unfreezing Layers: `param.requires_grad = True` (default for new layers).")

print("\nCommon Strategies:")
print("1. Feature Extraction (Frozen Backbone):")
print("   - Freeze all layers of the pre-trained backbone.")
print("   - Only train the newly added task-specific head.")
print("   - Faster, requires less data, good if downstream task is very different or data is scarce.")
print("   - Less powerful, as the backbone features are not adapted to the new task.")

print("2. Full Fine-tuning:")
print("   - All layers (backbone + new head) are unfrozen and trained.")
print("   - Adapts the entire model to the new task.")
print("   - Can achieve higher performance but requires more data and computation, and risks 'catastrophic forgetting'")
print("     if the new dataset is too small or learning rate too high.")

print("3. Partial Fine-tuning (Layer-wise Unfreezing):")
print("   - Initially, freeze the backbone and train only the head.")
print("   - Then, unfreeze some of the top layers of the backbone and train them along with the head.")
print("   - Gradually unfreeze more layers.")
print("   - A common approach to balance adaptation and stability.")

# --- Conceptual Example of Freezing ---
# if classification_model: # Assuming classification_model is defined and loaded
#     print("\nFreezing backbone layers (example):")
#     for name, param in classification_model.named_parameters():
#         if name.startswith("token_embedding") or \
#            name.startswith("positional_embedding") or \
#            name.startswith("transformer_blocks") or \
#            name.startswith("ln_final"):
#             param.requires_grad = False
#             # print(f"  Frozen: {name}")
#         else:
#             param.requires_grad = True # Ensure new head is trainable
#             # print(f"  Trainable: {name}")
#
#     # Verify which parameters are trainable
#     # trainable_params = sum(p.numel() for p in classification_model.parameters() if p.requires_grad)
#     # print(f"Number of trainable parameters after freezing: {trainable_params}")


# %% 5. Differential Learning Rates
print("\n--- 5. Differential Learning Rates ---\n")

print("It's often beneficial to use different learning rates for different parts of the model:")
print("- Pre-trained Backbone: Use a smaller learning rate. These layers already contain useful knowledge,")
print("  so you want to update them more cautiously to avoid disrupting this knowledge.")
print("- Newly Added Head: Use a larger learning rate. These layers are initialized randomly")
print("  and need to learn the task from scratch.")

# --- Conceptual Optimizer Setup for Differential LR ---
# if classification_model:
#     # Assume backbone params are those not in 'classification_head'
#     backbone_params = [p for name, p in classification_model.named_parameters()
#                        if p.requires_grad and not name.startswith("classification_head")]
#     head_params = [p for name, p in classification_model.named_parameters()
#                    if p.requires_grad and name.startswith("classification_head")]
#
#     if backbone_params and head_params:
#         optimizer = optim.AdamW([
#             {'params': backbone_params, 'lr': 1e-5}, # Small LR for backbone
#             {'params': head_params, 'lr': 1e-4}      # Larger LR for new head
#         ])
#         print("\nOptimizer set up with differential learning rates.")
#     elif head_params: # Only head is trainable (backbone frozen)
#          optimizer = optim.AdamW([{'params': head_params, 'lr': 1e-4}])
#          print("\nOptimizer set up for trainable head only.")
#     else:
#         print("No trainable parameters found for optimizer setup.")


# %% 6. Data Preparation for the Downstream Task
print("\n--- 6. Data Preparation for the Downstream Task ---\n")

print("You'll need a new `Dataset` and `DataLoader` for your specific fine-tuning task.")
print("The data format will depend on the task:")
print("- Text Classification: Pairs of (text_sequence, label_id).")
print("- Question Answering: Pairs of (context, question, answer_span).")
print("- Summarization: Pairs of (long_document, summary).")

print("\nTokenization should use the SAME tokenizer as the pre-trained model to ensure consistency.")
print("Input sequences might need to be formatted with special tokens (e.g., [CLS], [SEP])")
print("depending on the pre-trained model's architecture and how it was originally trained.")
print("(Our `SmallGPT` example doesn't explicitly use [CLS]/[SEP], but models like BERT do).")

# --- Conceptual Dataset for Sentiment Classification ---
# class SentimentDataset(Dataset):
#     def __init__(self, texts, labels, tokenizer, max_length):
#         self.texts = texts
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#
#     def __len__(self):
#         return len(self.texts)
#
#     def __getitem__(self, idx):
#         text = self.texts[idx]
#         label = self.labels[idx]
#         # Tokenize text - for BERT-like models, add special tokens
#         encoding = self.tokenizer.encode_plus(
#             text,
#             add_special_tokens=True, # e.g., [CLS], [SEP]
#             max_length=self.max_length,
#             padding='max_length', # Pad to max_length
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors='pt'
#         )
#         return {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(), # Needed if using padding
#             'labels': torch.tensor(label, dtype=torch.long)
#         }
#
# # Dummy data for example
# # fine_tune_texts = ["I love this movie!", "This book is terrible.", "It was an okay experience."]
# # fine_tune_labels = [2, 0, 1] # 0: neg, 1: neu, 2: pos
# # tokenizer_for_ft = AutoTokenizer.from_pretrained("gpt2") # Or the specific one used
# # fine_tune_dataset = SentimentDataset(fine_tune_texts, fine_tune_labels, tokenizer_for_ft, max_length=128)
# # fine_tune_loader = DataLoader(fine_tune_dataset, batch_size=16)


# %% 7. The Fine-tuning Loop
print("\n--- 7. The Fine-tuning Loop ---\n")

print("The fine-tuning loop is very similar to the pre-training loop (Module 6 & 8).")
print("Key differences:")
print("- Dataset & DataLoader: Use the new dataset for the downstream task.")
print("- Model: Use the modified model (with the new head).")
print("- Loss Function: Choose a loss appropriate for the downstream task.")
print("  - Classification: `nn.CrossEntropyLoss`.")
print("  - Regression: `nn.MSELoss`.")
print("- Optimizer: Might use differential learning rates.")
print("- Evaluation Metrics: Use metrics relevant to the downstream task (e.g., accuracy, F1-score for classification).")

# The structure of `train_one_epoch` and `evaluate` functions would remain largely the same,
# but they would operate on the fine-tuning data and use the task-specific loss and metrics.


# %% 8. Conceptual Introduction to Parameter-Efficient Fine-Tuning (PEFT)
print("\n--- 8. Conceptual Introduction to Parameter-Efficient Fine-Tuning (PEFT) ---\n")

print("Full fine-tuning of very large LLMs (billions of parameters) can still be computationally expensive")
print("and require significant GPU memory, even if less than training from scratch.")
print("Parameter-Efficient Fine-Tuning (PEFT) methods aim to address this by updating only a small")
print("subset of the model's parameters, or by adding a small number of new, trainable parameters.")

print("\nWhy PEFT?")
print("- Reduced Computational Cost & Memory: Training fewer parameters is faster and requires less memory.")
print("- Faster Iteration: Quicker to experiment with different tasks.")
print("- Avoids Catastrophic Forgetting: By keeping most of the pre-trained weights frozen, PEFT methods")
print("  are less prone to forgetting the general knowledge learned during pre-training.")
print("- Easier Deployment: Can have multiple small sets of PEFT weights for different tasks, all sharing")
print("  the same large frozen backbone model.")

print("\nExample PEFT Method: LoRA (Low-Rank Adaptation)")
print("LoRA's Core Idea:")
print("- For a pre-trained weight matrix W (e.g., in a Linear layer or attention projection), LoRA doesn't update W directly.")
print("- Instead, it learns two smaller, low-rank matrices A and B, such that the update to W is approximated by their product: ΔW = B @ A.")
print("- During fine-tuning, W is frozen, and only A and B are trained.")
print("- A and B have far fewer parameters than W (e.g., if W is d x d, A might be d x r and B r x d, where r << d).")
print("- For inference, the learned BA can be merged with W (W' = W + BA) so no extra latency is introduced.")
print("Libraries like Hugging Face `peft` provide easy implementations of LoRA and other PEFT methods.")


# %% 9. Connecting to Hugging Face `transformers`
print("\n--- 9. Connecting to Hugging Face `transformers` ---\n")

print("The Hugging Face `transformers` library heavily utilizes these fine-tuning concepts.")
print("- Loading Pre-trained Models: `AutoModel.from_pretrained('model_name')` loads a pre-trained backbone.")
print("- Task-Specific Heads: `AutoModelForSequenceClassification.from_pretrained('model_name', num_labels=N)`")
print("  loads a pre-trained model with a classification head suitable for N classes. The library handles")
print("  adding and initializing this head.")
print("- Fine-tuning with `Trainer` API: The `Trainer` class abstracts the training loop and handles many")
print("  aspects like data collation, optimizer setup, and evaluation. You provide datasets, model,")
print("  and training arguments.")
print("- PEFT Integration: The `peft` library integrates smoothly with `transformers` models, allowing you")
print("  to easily apply methods like LoRA.")

print("\nUnderstanding the underlying PyTorch concepts (nn.Module, optimizers, freezing layers)")
print("is crucial for effectively using and customizing Hugging Face `transformers`.")


# %% 10. Brief JAX Comparison
print("\n--- 10. Brief JAX Comparison ---\n")

print("Fine-tuning in JAX (e.g., with Flax or Haiku) follows similar principles but with a functional approach:")
print("- Model Loading: Load pre-trained parameters (often stored in a PyTree like a nested dictionary).")
print("- Model Modification: Define a new model function or Flax/Haiku module that incorporates the")
print("  pre-trained backbone parameters and adds new layers for the task-specific head.")
print("- Parameter Handling: Parameters for frozen parts are passed through without gradient updates.")
print("  Only parameters of trainable parts (new head, unfrozen backbone layers) are updated by the optimizer.")
print("- Optax for Optimizers: Optax provides flexible ways to define optimizers, including applying different")
print("  learning rates to different subsets of parameters (e.g., using `optax.multi_transform`).")
print("- Training Loop: The JIT-compiled training step function would take the full set of parameters,")
print("  compute gradients only for the trainable ones, and the optimizer would update them.")

print("\nThe core ideas of adapting a pre-trained model, choosing which parts to train, and preparing")
print("task-specific data remain consistent across frameworks.")


# %% Conclusion
print("\n--- Module 12 Summary ---\n")
print("Key Takeaways:")
print("- Fine-tuning adapts powerful pre-trained models to specific downstream tasks, saving resources and often yielding better results.")
print("- Key steps include modifying the model head, deciding which layers to freeze/unfreeze, and potentially using differential learning rates.")
print("- PEFT methods like LoRA offer efficient alternatives to full fine-tuning for very large models.")
print("- Data preparation and the training loop structure are adapted for the new task's requirements.")
print("- These concepts are fundamental to working effectively with libraries like Hugging Face `transformers`.")

print("\nThis concludes our core curriculum on PyTorch for LLMs! Further study could involve")
print("deeper dives into specific PEFT methods, advanced distributed training (FSDP),")
print("or specialized LLM architectures and evaluation techniques.")

print("\nEnd of Module 12.")