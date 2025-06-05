# 17_understanding_model_repository_files.py
# This script explains the common files found in open-source model repositories,
# such as those on the Hugging Face Hub or GitHub.
# Understanding these files is crucial for effectively using, adapting,
# and contributing to open-source AI models.

# Overview:
# 1. General explanation of typical files and their purposes.
#    - README.md
#    - Configuration files (e.g., config.json)
#    - Model Architecture Code (e.g., model.py)
#    - Model weights (e.g., .safetensors, .bin)
#    - Tokenizer files (e.g., tokenizer.json, vocab.txt, merges.txt)
#    - Generation configuration (e.g., generation_config.json)
#    - .gitignore
#    - requirements.txt / environment files
#    - Training, inference, and evaluation scripts
#    - LICENSE
# 2. Illustration using a real-world example:
#    - Downloading and inspecting files from "HuggingFaceTB/SmolLM2-135M-Instruct"
# 3. Deep dive into the .safetensors format
# 4. Reference to a code-centric repository (e.g., nanoVLM on GitHub)

import json
from huggingface_hub import hf_hub_download
import os
from safetensors import safe_open # For inspecting .safetensors files
import struct # For unpacking binary data (the header length)

print("=" * 60)
print("Understanding Common Files in Model Repositories")
print("=" * 60)
print("""
When you explore an open-source model, whether on the Hugging Face Hub,
GitHub, or another platform, you'll typically find a collection of files.
These files provide everything needed to understand, load, and use the model,
and sometimes to retrain or fine-tune it. Let's go through the most common ones.
""")

# ------------------------------------------------------------------------------
print("\n" + "-" * 50)
print("1. README.md")
print("-" * 50)
print("""
Purpose:
The `README.md` (Markdown file) is the entry point to understanding the model.
It usually contains:
- A description of the model: What it is, what it does.
- Model capabilities and limitations.
- How to use the model (code examples for inference).
- Information about training data and procedures.
- Evaluation results and benchmarks.
- Citation information (if you use the model in your work).
- Licensing information.
- Ethical considerations and intended uses.

Importance:
Essential. This is the first file you should read. A good README saves you
a lot of time and helps you decide if the model is right for your needs.
If a model provider shares only weights without a README, it's much harder
to understand and use the model correctly.
""")

# ------------------------------------------------------------------------------
print("\n" + "-" * 50)
print("2. Model Architecture Code (e.g., model.py, Python files)")
print("-" * 50)
print("""
Purpose:
These are the Python files (`.py`) that define the actual structure of the
neural network (e.g., the `torch.nn.Module` classes in PyTorch, or
`tf.keras.Model` classes in TensorFlow). This is the code that implements
the layers, connections, and forward pass logic of the model.

Importance:
Absolutely essential. The model weights are just numbers; the architecture code
provides the framework into which these numbers are loaded to form a functional model.

Where to find it:
- For standard architectures (e.g., BERT, GPT-2, LLaMA, T5, GPT-NeoX):
  The code is often part of a major library like Hugging Face `transformers`.
  You don't need separate `.py` files in the model repo itself because the
  library provides them. The `config.json` (specifically the `model_type`
  field) tells the library which of its internal architecture classes to use.
- For custom models or research projects not yet in standard libraries:
  These `.py` files MUST be provided by the authors, usually in a linked
  GitHub repository (e.g., in a `models/` directory or a main `model.py` file).
  Without them, you cannot instantiate the model to load the weights.
  Repositories like `huggingface/nanoVLM` explicitly provide this in their
  `models/` directory.

If you only have weights and a `config.json` for a custom architecture without
the Python code defining that architecture, the model is unusable.
""")

# ------------------------------------------------------------------------------
print("\n" + "-" * 50)
print("3. Model Configuration File (e.g., config.json)")
print("-" * 50)
print("""
Purpose:
This file (commonly `config.json` for Hugging Face Transformers models) stores
the blueprint or architecture *parameters* of the model. It contains
hyperparameters and settings that configure the model architecture defined
in the Python code (see point 2). Examples:
- `model_type`: (e.g., 'gpt_neox', 'bert', 'llama') - This often links to a
  specific architecture class within a library like `transformers`.
- Number of layers, attention heads, hidden size.
- Vocabulary size.
- Activation functions.
- Dropout rates.
- Any specific architectural choices that are parameterized.

Importance:
Crucial. It tells the system (either a library like `transformers` or custom
model code) how to instantiate the model architecture with specific dimensions
and settings before loading the weights.
""")

# ------------------------------------------------------------------------------
print("\n" + "-" * 50)
print("4. Model Weights File (e.g., model.safetensors, pytorch_model.bin)")
print("-" * 50)
print("""
Purpose:
These files contain the actual learned parameters (weights and biases) of the
neural network. This is the "intelligence" of the model.
- `model.safetensors`: A newer, safer, and often faster format for storing tensors.
  It's generally preferred as it's not subject to the same security risks as
  Python's pickle format (used in `.bin` files). It contains a JSON header
  describing the tensors, followed by the raw binary tensor data.
- `pytorch_model.bin` (or `tf_model.h5` for TensorFlow): The traditional format,
  often using Python's pickle for PyTorch models.

Importance:
The absolute core. Without the weights, you only have an uninitialized model
architecture. These are the files that are typically the largest in a model repo.
Sometimes, for very large models, weights are sharded into multiple files
(e.g., `pytorch_model-00001-of-00002.bin`).
We will inspect the structure of a .safetensors file in more detail later.
""")

# ------------------------------------------------------------------------------
print("\n" + "-" * 50)
print("5. Tokenizer Files")
print("-" * 50)
print("""
Purpose:
Language models operate on numerical representations of text (token IDs).
Tokenizer files define how to convert raw text into these token IDs and vice-versa.
Common files include:
- `tokenizer.json`: Often a single, comprehensive file containing the entire
  tokenizer state: vocabulary, merge rules (for BPE/WordPiece), special tokens,
  and configuration. This is common for tokenizers from the `tokenizers` library.
- `tokenizer_config.json`: Contains settings for loading the tokenizer, like the
  tokenizer class, paths to other files (if not all in one `tokenizer.json`),
  and settings for special tokens (e.g., BOS token, EOS token, PAD token).
- `vocab.json` or `vocab.txt`: The mapping from tokens (words, subwords) to integer IDs.
- `merges.txt`: For BPE (Byte Pair Encoding) tokenizers, this file contains the
  learned merge rules.
- `special_tokens_map.json`: Defines special tokens used by the model (e.g.,
  `<s>`, `</s>`, `[CLS]`, `[SEP]`, `[PAD]`, `[UNK]`) and their string representations.

Importance:
Essential for any NLP model. You need the *exact* tokenizer used during training
to get meaningful results during inference. Using a different tokenizer will lead
to a mismatch between how text is processed and what the model expects, resulting
in poor performance or errors.
""")

# ------------------------------------------------------------------------------
print("\n" + "-" * 50)
print("6. Generation Configuration (e.g., generation_config.json)")
print("-" * 50)
print("""
Purpose:
For generative models (like GPT), this file stores default parameters for
controlling how text is generated. Examples include:
- `max_length` or `max_new_tokens`: Maximum length of the generated sequence.
- `num_beams`: For beam search.
- `do_sample`: Whether to use sampling (True/False).
- `temperature`: Controls randomness in sampling (lower is more deterministic).
- `top_k`, `top_p`: For nucleus or top-k sampling.
- IDs for special tokens like `pad_token_id`, `eos_token_id`, `bos_token_id`.

Importance:
Convenient. While these parameters can usually be overridden in your inference
code, this file provides sensible defaults that the model authors found to work well.
It helps in achieving reproducible generation results.
""")

# ------------------------------------------------------------------------------
print("\n" + "-" * 50)
print("7. `.gitignore`")
print("-" * 50)
print("""
Purpose:
This is a file used by the Git version control system. It specifies intentionally
untracked files that Git should ignore. This typically includes:
- Local environment files (e.g., `.venv/`, `__pycache__/`).
- Large data files or downloaded datasets not meant to be in the repo.
- Log files, temporary build artifacts.
- API keys or sensitive credentials (though these should ideally be managed
  through other means like environment variables or .env files which are also gitignored).

Importance:
Mainly for developers contributing to or managing the model's source code repository.
It keeps the repository clean, focused on essential files, and prevents accidental
commits of large or unnecessary files. You'll find this in GitHub repos more often
than directly in Hugging Face Hub model repos (which focus on model artifacts).
""")

# ------------------------------------------------------------------------------
print("\n" + "-" * 50)
print("8. `requirements.txt` / Environment Files (e.g., `pyproject.toml`)")
print("-" * 50)
print("""
Purpose:
These files list the Python libraries and their versions required to run the
model's code (e.g., training scripts, inference examples).
- `requirements.txt`: A common format listing package dependencies.
- `pyproject.toml` (with Poetry or PDM) or `setup.py`/`setup.cfg`: More modern
  ways to manage dependencies and package a Python project.

Importance:
Crucial for reproducibility. To ensure that the code runs correctly, you need
to have the same or compatible versions of libraries (like `torch`, `transformers`,
`datasets`, etc.) that were used to develop and test the model.
Like `.gitignore`, these are more common in source code repositories (e.g., on GitHub)
that accompany a model, rather than being a primary artifact in a model-only Hub repo.
""")

# ------------------------------------------------------------------------------
print("\n" + "-" * 50)
print("9. Training, Inference, and Evaluation Scripts (e.g., .py files)")
print("-" * 50)
print("""
Purpose:
These are Python scripts (or notebooks) that show how to:
- `train.py`: Train or fine-tune the model. This might include data preprocessing,
  model setup, training loop, and saving checkpoints.
- `generate.py` or `inference.py`: Run the model to make predictions or generate text.
- `evaluate.py`: Evaluate the model's performance on benchmark datasets.

Importance:
Highly valuable for understanding how the model was built and how to use it
effectively. They provide practical examples and can be starting points for
your own projects. Repositories like `nanoVLM` (which we'll mention later)
are good examples of this, where the code IS the main product.
If these are missing, you rely solely on the README or library abstractions
(like Hugging Face `pipeline`) to use the model.
""")

# ------------------------------------------------------------------------------
print("\n" + "-" * 50)
print("10. LICENSE File")
print("-" * 50)
print("""
Purpose:
This file (e.g., `LICENSE`, `LICENSE.md`) specifies the legal terms under
which the model, its weights, and associated code can be used, modified,
and distributed. Common licenses include Apache 2.0, MIT, CC-BY-SA, etc.
Some models might have custom licenses or specific use restrictions.

Importance:
Critical. Always check the license before using a model, especially for
commercial purposes or if you plan to redistribute it or create derivative works.
The license dictates what you are allowed and not allowed to do.
Open-source doesn't always mean "free for any use."
""")

# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Illustrating with HuggingFaceTB/SmolLM2-135M-Instruct")
print("=" * 60)
print("""
Now, let's download and inspect some of these files from a real model on the
Hugging Face Hub: "HuggingFaceTB/SmolLM2-135M-Instruct".
We'll use the `huggingface_hub` library for this.
""")

repo_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
# Create a local directory to store downloaded files
local_dir = "smolLM2_example_files"
os.makedirs(local_dir, exist_ok=True)
print(f"Will download files to: ./{local_dir}/\n")

# --- README.md ---
print("\n--- Inspecting README.md ---")
try:
    readme_path = hf_hub_download(repo_id=repo_id, filename="README.md", local_dir=local_dir, local_dir_use_symlinks=False)
    print(f"README.md downloaded to: {readme_path}")
    with open(readme_path, 'r', encoding='utf-8') as f:
        readme_content = f.read()
    print("\nFirst 500 characters of README.md:")
    print(readme_content[:500] + "...")
except Exception as e:
    print(f"Could not download or read README.md: {e}")

# --- config.json ---
print("\n--- Inspecting config.json ---")
try:
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json", local_dir=local_dir, local_dir_use_symlinks=False)
    print(f"config.json downloaded to: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    print("\nContent of config.json (some key-value pairs):")
    for key, value in list(config_data.items())[:7]: # Print first 7 key-value pairs
        print(f"  {key}: {value}")
    if len(config_data) > 7:
        print("  ...")
    print(f"  Model type: {config_data.get('model_type', 'N/A')}")
    print(f"  Vocab size: {config_data.get('vocab_size', 'N/A')}")
    print(f"  Hidden size: {config_data.get('hidden_size', config_data.get('n_embd', 'N/A'))}") # n_embd for some GPT-like models
    print(f"  Num hidden layers: {config_data.get('num_hidden_layers', config_data.get('n_layer', 'N/A'))}")
except Exception as e:
    print(f"Could not download or read config.json: {e}")

# --- tokenizer.json ---
print("\n--- Inspecting tokenizer.json ---")
try:
    tokenizer_json_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json", local_dir=local_dir, local_dir_use_symlinks=False)
    print(f"tokenizer.json downloaded to: {tokenizer_json_path}")
    with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    print("\nContent of tokenizer.json (structure and some details):")
    print(f"  Keys: {list(tokenizer_data.keys())}")
    if 'model' in tokenizer_data:
        print(f"  Model type from tokenizer: {tokenizer_data['model'].get('type')}")
        print(f"  Vocab size from tokenizer: {len(tokenizer_data['model'].get('vocab', {}))}")
    if 'added_tokens' in tokenizer_data:
        print(f"  Number of added special tokens: {len(tokenizer_data['added_tokens'])}")
        if tokenizer_data['added_tokens']:
             print(f"  Example added token: {tokenizer_data['added_tokens'][0]}")
    print("  (This file can be quite large and complex, containing the full vocabulary, merge rules, etc.)")
except Exception as e:
    print(f"Could not download or read tokenizer.json: {e}")

# --- tokenizer_config.json ---
print("\n--- Inspecting tokenizer_config.json ---")
try:
    tokenizer_config_path = hf_hub_download(repo_id=repo_id, filename="tokenizer_config.json", local_dir=local_dir, local_dir_use_symlinks=False)
    print(f"tokenizer_config.json downloaded to: {tokenizer_config_path}")
    with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
        tokenizer_config_data = json.load(f)
    print("\nContent of tokenizer_config.json:")
    for key, value in tokenizer_config_data.items():
        print(f"  {key}: {value}")
    print(f"  Tokenizer class hint: {tokenizer_config_data.get('tokenizer_class', 'N/A')}")
except Exception as e:
    print(f"Could not download or read tokenizer_config.json: {e}")

# --- special_tokens_map.json ---
print("\n--- Inspecting special_tokens_map.json ---")
try:
    special_tokens_path = hf_hub_download(repo_id=repo_id, filename="special_tokens_map.json", local_dir=local_dir, local_dir_use_symlinks=False)
    print(f"special_tokens_map.json downloaded to: {special_tokens_path}")
    with open(special_tokens_path, 'r', encoding='utf-8') as f:
        special_tokens_data = json.load(f)
    print("\nContent of special_tokens_map.json:")
    for key, value in special_tokens_data.items():
        print(f"  {key}: {value}")
except Exception as e:
    print(f"Could not download or read special_tokens_map.json: {e}")

# --- generation_config.json ---
print("\n--- Inspecting generation_config.json ---")
try:
    generation_config_path = hf_hub_download(repo_id=repo_id, filename="generation_config.json", local_dir=local_dir, local_dir_use_symlinks=False)
    print(f"generation_config.json downloaded to: {generation_config_path}")
    with open(generation_config_path, 'r', encoding='utf-8') as f:
        generation_config_data = json.load(f)
    print("\nContent of generation_config.json (some key-value pairs):")
    for key, value in list(generation_config_data.items())[:7]:
        print(f"  {key}: {value}")
    if len(generation_config_data) > 7:
        print("  ...")
    print(f"  Max new tokens (example): {generation_config_data.get('max_new_tokens', 'N/A')}")
    print(f"  Do sample (example): {generation_config_data.get('do_sample', 'N/A')}")
except Exception as e:
    print(f"Could not download or read generation_config.json: {e}")


# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. Deep Dive: Inspecting model.safetensors")
print("=" * 60)
print("""
The `model.safetensors` file is crucial as it holds the model's learned weights.
It's a binary format that contains:
1. An 8-byte header indicating the length of the JSON metadata.
2. The JSON metadata: This describes each tensor (name, data type, shape, and its byte offset in the file).
3. Raw tensor data: The actual numerical values of the weights, stored in binary.

Let's download 'model.safetensors' for SmolLM2 and inspect its JSON header by manually parsing it.
This approach is robust across different versions of the 'safetensors' library.
""")

safetensors_file_name = "model.safetensors"
# We still need these for other parts of the script if they were used with safe_open
# from safetensors import safe_open # Not strictly needed for manual header parsing below

try:
    safetensors_path = hf_hub_download(
        repo_id=repo_id,
        filename=safetensors_file_name,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    print(f"'{safetensors_file_name}' downloaded to: {safetensors_path}")

    header_data = {}
    with open(safetensors_path, "rb") as f_raw:
        # 1. Read the header length (8 bytes, little-endian uint64)
        header_len_bytes = f_raw.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError("Could not read safetensors header length. File might be too small or corrupted.")
        header_len = struct.unpack('<Q', header_len_bytes)[0]

        # 2. Read the JSON header (n bytes, UTF-8 encoded)
        json_header_bytes = f_raw.read(header_len)
        if len(json_header_bytes) < header_len:
            raise ValueError("Could not read the full JSON header from safetensors. File might be corrupted.")
        json_header_str = json_header_bytes.decode('utf-8')
        
        # 3. Parse the JSON header
        header_data = json.loads(json_header_str)

    print("\n--- Parsed JSON Header from model.safetensors (manual parsing) ---")
    
    global_metadata = None
    if "__metadata__" in header_data:
        # This is global metadata, not tensor-specific. Pop it so it's not counted as a tensor.
        global_metadata = header_data.pop("__metadata__")
        print(f"Global metadata found: {global_metadata}")

    tensor_keys = list(header_data.keys())
    tensor_count = len(tensor_keys)
    print(f"Found {tensor_count} tensors described in the header.")

    # Let's look at the metadata for the first few tensors
    print("\nMetadata for the first up to 5 tensors:")
    for i, key in enumerate(tensor_keys[:5]):
        tensor_info = header_data[key]
        print(f"\nTensor name: {key}")
        print(f"  dtype: {tensor_info.get('dtype', 'N/A')}")
        print(f"  shape: {tensor_info.get('shape', 'N/A')}")
        print(f"  data_offsets (start_byte, end_byte): {tensor_info.get('data_offsets', 'N/A')}")
    
    if tensor_count > 5:
        print("\n... and more tensors ...")

    print("\n--- Summary of .safetensors content ---")
    print("The .safetensors file's initial part is a JSON header (as printed above for each tensor name).")
    print("This JSON describes all tensors: their names, data types, shapes, and where their binary data is located ('data_offsets').")
    print("The remainder of the file consists of the concatenated raw binary data for all these tensors.")
    print("This structure is efficient for loading and safe because it avoids arbitrary code execution (unlike Python's pickle).")
    print("For actually loading these tensors into a model, you would use the 'safetensors' library, e.g., `with safe_open(...)`.")

except Exception as e:
    print(f"Could not download or inspect '{safetensors_file_name}': {e}")
    print("If the error is related to 'safetensors', consider updating it: `pip install --upgrade safetensors`")
    print("The manual parsing method attempted here should generally work if the file is a valid .safetensors file.")


# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. Reference to Code-Centric Repositories (e.g., GitHub)")
print("=" * 60)
print("""
While Hugging Face Hub model pages are excellent for hosting model artifacts
(weights, configs, tokenizer files), full projects often live on GitHub.
A repository like `huggingface/nanoVLM` (which you mentioned) is a good example.
In such repositories, you'd typically find:

- Python scripts for the model definition (e.g., `models/vision_language_model.py` in nanoVLM).
- Training scripts (`train.py`).
- Inference/generation scripts (`generate.py`).
- Evaluation scripts or utilities (`benchmark_suite.py`).
- A `requirements.txt` or similar to set up the environment.
- A `.gitignore` file.
- Detailed `README.md` explaining setup, training, and usage.
- Sometimes, example data or scripts to download data (`data/` directory).

These code repositories provide the full context for how a model is built,
trained, and intended to be used, complementing the model artifacts hosted
on the Hub. For instance, the `nanoVLM` README links to Hub models for its
pretrained weights but the GitHub repo contains all the code to work with them.
""")

# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Conclusion")
print("=" * 60)
print("""
Understanding these common files is key to navigating the open-source AI landscape.
When you find a model:
1. Start with the `README.md`.
2. Check the `LICENSE`.
3. Look at `config.json` to understand the model architecture.
4. Ensure you have the correct `tokenizer` files.
5. The model weights (`.safetensors` or `.bin`) are what you'll load. The `.safetensors`
   format includes a JSON header describing tensor metadata, followed by binary data.
6. `generation_config.json` can provide good starting points for inference.
7. If there's an associated GitHub repo, explore it for training code,
   `requirements.txt`, and deeper insights.

If critical files like weights, config, or tokenizer files are missing,
or if the README is uninformative, it will be very difficult to make
productive use of the model. Well-documented and complete repositories
are a hallmark of good open-source practice.
""")

print(f"\nExample files from {repo_id} (including model.safetensors) were downloaded to the '{local_dir}' directory.")
print("You can inspect them further there.")
