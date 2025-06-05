# -*- coding: utf-8 -*-
# 05_data_loading.py

# Module 5: Data Loading and Processing with `torch.utils.data`
#
# This script covers:
# 1. `torch.utils.data.Dataset`: Abstracting dataset access.
# 2. Creating Custom Datasets: Implementing `__len__` and `__getitem__`.
# 3. Tokenization & Numericalization: Converting text to numbers models understand.
# 4. `torch.utils.data.DataLoader`: Batching, shuffling, and parallel loading.
# 5. Handling Variable Lengths: Implementing custom `collate_fn` for padding.
# 6. Comparisons to JAX data loading approaches.

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence # Useful for padding sequences

print("--- Module 5: Data Loading and Processing with torch.utils.data ---\n")
print(f"Using PyTorch version: {torch.__version__}")


# %% 1. `torch.utils.data.Dataset`
print("\n--- 1. `torch.utils.data.Dataset` ---\n")

# A `Dataset` is an object that represents your data source. It needs two methods:
# - `__len__(self)`: Should return the total number of samples in the dataset.
# - `__getitem__(self, idx)`: Should return the sample at the given index `idx`.
#
# PyTorch provides built-in datasets (e.g., for computer vision), but you often
# need to create custom ones for your specific data.

# *** SWE Note ***
# Well-designed `Dataset` classes are crucial for good software engineering.
# They encapsulate all logic related to accessing and preprocessing individual
# data points (e.g., reading files, cleaning text, applying transformations).
# This separates data concerns from model training logic, making the codebase
# cleaner, more modular, and easier to test and reuse.


# %% 2. Creating a Custom Dataset
print("\n--- 2. Creating a Custom Dataset ---\n")

# Let's imagine a simple dataset for translation (English -> French)
# Sentences have variable lengths.
raw_data = [
    ("Hello world", "Bonjour le monde"),
    ("How are you?", "Comment allez-vous?"),
    ("Machine learning is fun", "L'apprentissage automatique est amusant"),
    ("PyTorch", "PyTorch"), # Short example
]

class SimpleTranslationDataset(Dataset):
    """A simple custom dataset for English-French sentence pairs."""
    def __init__(self, data):
        super().__init__()
        # In a real scenario, 'data' might be file paths or loaded data structures
        self.data = data
        print(f"Initialized dataset with {len(self.data)} samples.")

    def __len__(self):
        # Returns the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Returns the sample pair at the given index
        # In this simple case, it's just accessing the list.
        # In real datasets, this might involve reading files, decoding images, etc.
        english_text, french_text = self.data[idx]
        # Return as a dictionary for clarity (often good practice)
        return {"english": english_text, "french": french_text}

# Instantiate the dataset
translation_dataset = SimpleTranslationDataset(raw_data)

# Demonstrate accessing elements and length
print(f"Total dataset size: {len(translation_dataset)}")
print(f"Sample at index 0: {translation_dataset[0]}")
print(f"Sample at index 2: {translation_dataset[2]}")


# %% 3. Tokenization and Numericalization
print("\n--- 3. Tokenization and Numericalization ---\n")

# Models need numerical input, not raw text. This involves:
# 1. Tokenization: Splitting text into units (tokens) - words, subwords, or characters.
# 2. Numericalization: Mapping tokens to integer indices based on a vocabulary.
# 3. Adding Special Tokens: Like <PAD> (padding), <SOS> (start), <EOS> (end).

# Let's use simple character-level tokenization for demonstration.
# In practice (especially for LLMs), you'd use more advanced tokenizers
# like SentencePiece, BPE (used by GPT/Llama), or WordPiece (used by BERT),
# often via libraries like Hugging Face `tokenizers`.

# --- Build Vocabulary (Character Level) ---
all_chars = set()
for eng, fra in raw_data:
    all_chars.update(eng)
    all_chars.update(fra)

# Define special tokens
PAD_TOKEN = "<PAD>" # Padding
SOS_TOKEN = "<SOS>" # Start of Sentence
EOS_TOKEN = "<EOS>" # End of Sentence
UNK_TOKEN = "<UNK>" # Unknown token (optional, less needed for char level)
special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

# Create mappings
# Leave index 0 for padding usually
vocab = {char: i+len(special_tokens) for i, char in enumerate(sorted(list(all_chars)))}
# Add special tokens to the beginning of the vocab
for i, token in enumerate(special_tokens):
    vocab[token] = i

# Inverse mapping for decoding (optional but useful)
idx_to_char = {i: char for char, i in vocab.items()}

vocab_size = len(vocab)
PAD_IDX = vocab[PAD_TOKEN]
SOS_IDX = vocab[SOS_TOKEN]
EOS_IDX = vocab[EOS_TOKEN]
UNK_IDX = vocab[UNK_TOKEN]

print(f"Built character vocabulary with {vocab_size} items.")
print(f"  Padding index: {PAD_IDX}")
# print(f"  Vocab sample: {list(vocab.items())[:10]}...") # Print some vocab items


# --- Modify Dataset to Return Numericalized Tensors ---
class NumericalizedTranslationDataset(Dataset):
    """Dataset returning numericalized sentence pairs with special tokens."""
    def __init__(self, data, vocab):
        super().__init__()
        self.data = data
        self.vocab = vocab
        self.pad_idx = self.vocab[PAD_TOKEN]
        self.sos_idx = self.vocab[SOS_TOKEN]
        self.eos_idx = self.vocab[EOS_TOKEN]
        self.unk_idx = self.vocab[UNK_TOKEN]
        print("Initialized numericalized dataset.")

    def __len__(self):
        return len(self.data)

    def _numericalize(self, text):
        # Convert string to list of indices, adding SOS/EOS
        tokens = list(text) # Character-level tokens
        indices = [self.vocab.get(token, self.unk_idx) for token in tokens]
        return torch.tensor([self.sos_idx] + indices + [self.eos_idx], dtype=torch.long)

    def __getitem__(self, idx):
        english_text, french_text = self.data[idx]
        eng_tensor = self._numericalize(english_text)
        fra_tensor = self._numericalize(french_text)
        return {"english": eng_tensor, "french": fra_tensor}

# Instantiate the numericalized dataset
num_translation_dataset = NumericalizedTranslationDataset(raw_data, vocab)
print(f"Numericalized sample at index 1:\n{num_translation_dataset[1]}")
print(f"  English tensor shape: {num_translation_dataset[1]['english'].shape}")
print(f"  French tensor shape: {num_translation_dataset[1]['french'].shape}") # Note the different lengths!


# %% 4. `torch.utils.data.DataLoader`
print("\n--- 4. `torch.utils.data.DataLoader` ---\n")

# DataLoader wraps a Dataset and provides an iterator for easy batching, shuffling,
# and optional parallel data loading using multiple workers.

# Key Arguments:
# - `dataset`: The Dataset instance.
# - `batch_size`: How many samples per batch.
# - `shuffle`: Set to `True` to shuffle data at every epoch (crucial for training).
# - `num_workers`: Number of subprocesses to use for data loading. `0` means data
#   is loaded in the main process. Increasing this can speed up data loading,
#   but uses more CPU/RAM. Can have issues on Windows (see note below).
# - `collate_fn`: A function that merges a list of samples into a mini-batch tensor.

# Try DataLoader without a custom collate_fn first - likely problematic for variable lengths.
simple_loader = DataLoader(num_translation_dataset, batch_size=2, shuffle=False)

print("Iterating through DataLoader without custom collate_fn:")
# The following loop is commented out because it is expected to raise a
# RuntimeError. The default collate function tries to stack tensors from
# individual samples into a batch. Since our samples (sentences) have
# varying lengths, the resulting tensors have different shapes (e.g., [13] and [14]),
# which cannot be stacked directly. This demonstrates the need for a custom
# collate_fn that handles padding (shown in the next section).
#
# for i, batch in enumerate(simple_loader):
#     print(f" Batch {i}:")
#     # The batch is likely a dictionary where values are lists of tensors of varying lengths
#     print(f"  Type of batch['english']: {type(batch['english'])}")
#     print(f"  Length of batch['english'] list: {len(batch['english'])}")
#     print(f"  Shape of first english tensor in list: {batch['english'][0].shape}")
#     print(f"  Shape of second english tensor in list: {batch['english'][1].shape}")
#     if i == 0:
#         break # Only show first batch

print("(Skipping iteration with default collate_fn as it causes RuntimeError with variable lengths)")

# This structure (list of tensors, if it didn't error) is generally not directly usable by models,
# which expect rectangular batch tensors. We need `collate_fn`.


# %% 5. Handling Variable Lengths: Custom `collate_fn`
print("\n--- 5. Handling Variable Lengths: Custom `collate_fn` ---\n")

# A custom `collate_fn` takes a list of samples (as returned by `Dataset.__getitem__`)
# and bundles them into a single batch, usually involving padding.

def padding_collate_fn(batch, pad_idx):
    """
    Collates a list of samples (dictionaries with 'english' and 'french' tensors)
    into padded batch tensors.

    Args:
        batch (list): A list of dictionaries, e.g., [{'english': tensor1, 'french': tensorA}, ...]
        pad_idx (int): The index to use for padding.

    Returns:
        dict: A dictionary containing padded batch tensors, e.g.,
              {'english': padded_eng_batch, 'french': padded_fra_batch}
    """
    # Separate the English and French tensors from the list of dictionaries
    eng_tensors = [item['english'] for item in batch]
    fra_tensors = [item['french'] for item in batch]

    # Use pad_sequence to pad tensors in each list to the max length in that list
    # pad_sequence expects a list of tensors and pads them to the longest tensor in the list.
    # batch_first=True makes the output shape (batch_size, max_seq_len)
    eng_padded = pad_sequence(eng_tensors, batch_first=True, padding_value=pad_idx)
    fra_padded = pad_sequence(fra_tensors, batch_first=True, padding_value=pad_idx)

    return {'english': eng_padded, 'french': fra_padded}

# We need to pass pad_idx to the collate function. DataLoader doesn't directly support
# passing extra args to collate_fn, so we use functools.partial or a lambda.
from functools import partial

# Note on `num_workers`:
# Using num_workers > 0 on Windows requires the data loading loop to be inside
# the `if __name__ == '__main__':` block. On Linux/macOS it's generally fine.
# For simplicity here, we'll use num_workers=0.
num_workers = 0
# If you were running on Windows and wanted num_workers > 0, you'd need:
# if __name__ == '__main__':
#     loader = DataLoader(...)
#     for batch in loader: ...

padded_loader = DataLoader(
    num_translation_dataset,
    batch_size=2,
    shuffle=False, # Keep false for predictable output demonstration
    num_workers=num_workers,
    collate_fn=partial(padding_collate_fn, pad_idx=PAD_IDX) # Use partial to pass pad_idx
    # Alternative using lambda:
    # collate_fn=lambda batch: padding_collate_fn(batch, pad_idx=PAD_IDX)
)

print("\nIterating through DataLoader WITH custom padding_collate_fn:")
for i, batch in enumerate(padded_loader):
    print(f" Batch {i}:")
    # Now the batch tensors should be rectangular
    print(f"  Type of batch['english']: {type(batch['english'])}")
    print(f"  Shape of padded 'english' batch tensor: {batch['english'].shape}")
    print(f"  Padded 'english' batch tensor:\n{batch['english']}")
    print(f"  Shape of padded 'french' batch tensor: {batch['french'].shape}")
    # print(f"  Padded 'french' batch tensor:\n{batch['french']}")
    if i == 0: break # Only show first batch

print("\nNow the batches are properly padded tensors, ready for model input!")


# %% 6. JAX Comparison
print("\n--- 6. JAX Comparison ---\n")

print("JAX itself does not include high-level data loading utilities like `Dataset` or `DataLoader`.")
print("Common approaches in the JAX ecosystem:")
print("- Using `tf.data`: TensorFlow's data loading library is feature-rich, performs well,")
print("  and integrates reasonably well (can output NumPy arrays which JAX consumes). Often used in Google projects.")
print("- Using Hugging Face `datasets`: This library is framework-agnostic and can load/process")
print("  many datasets. It can output data in formats consumable by JAX (e.g., NumPy).")
print("- Custom Python Generators/Iterators: Writing standard Python generators (`yield` batches)")
print("  that perform loading, preprocessing, and batching manually or using libraries like NumPy.")
print("- Third-party libraries: Libraries aiming to bridge this gap sometimes emerge.")

print("\nPros/Cons:")
print("- PyTorch (`Dataset`/`DataLoader`): Tightly integrated, convenient standard within the ecosystem,")
print("  good multiprocessing support (`num_workers`).")
print("- JAX (External Tools): Requires choosing and integrating external libraries (e.g., `tf.data`, HF `datasets`).")
print("  Might offer different performance characteristics or features depending on the chosen library.")
print("  Parallel data loading needs to be handled by the chosen library or custom implementation.")


# %% Conclusion
print("\n--- Module 5 Summary ---\n")
print("Key Takeaways:")
print("- `torch.utils.data.Dataset` provides a standard way to represent datasets (`__len__`, `__getitem__`).")
print("- `torch.utils.data.DataLoader` provides efficient batching, shuffling, and parallel loading.")
print("- Text data requires tokenization and numericalization before being fed to models.")
print("- Custom `collate_fn` is essential for handling variable-length sequences by padding them into batch tensors.")
print("- PyTorch offers an integrated data loading solution, while JAX typically relies on external libraries or custom implementations.")

print("\nEnd of Module 5.")