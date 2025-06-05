# -*- coding: utf-8 -*-
# 10_swe_practices_project_structure.py

# Module 10: SWE Practices & Project Structure for PyTorch
#
# This script discusses essential software engineering practices and how to
# structure larger PyTorch projects for maintainability, collaboration, and
# reproducibility. While smaller scripts are fine for learning, real-world
# projects benefit greatly from good organization.
#
# We will cover:
# 1. Why SWE Practices Matter in ML.
# 2. Typical Project Directory Layout.
# 3. Configuration Management (argparse, Hydra/YAML).
# 4. Code Modularity and Reusability.
# 5. Type Hinting.
# 6. Logging.
# 7. Unit Testing (Conceptual Overview).
# 8. Dependency Management.

import argparse
import logging
from typing import List, Dict, Tuple, Optional # For demonstrating type hints
import torch
import torch.nn as nn

print("--- Module 10: SWE Practices & Project Structure for PyTorch ---\n")


# %% 1. Why SWE Practices Matter in ML
print("\n--- 1. Why SWE Practices Matter in ML ---\n")

print("Machine learning projects, especially those involving deep learning models like LLMs,")
print("can become complex quickly. Adopting good software engineering practices helps to:")
print("- Improve Code Readability & Maintainability: Makes it easier for you and others")
print("  to understand, modify, and debug the code.")
print("- Enhance Collaboration: Standardized structure and clear code allow teams to work")
print("  more effectively.")
print("- Ensure Reproducibility: Consistent setups, configuration management, and versioning")
print("  are key to reproducing experiments and results.")
print("- Facilitate Scalability: Well-structured code is easier to scale up for larger datasets,")
print("  more complex models, or deployment.")
print("- Reduce Bugs: Modular design, testing, and clear interfaces can help catch errors early.")
print("- Streamline Experimentation: Easily change configurations, swap model components, or")
print("  run different training regimes.")


# %% 2. Typical Project Directory Layout
print("\n--- 2. Typical Project Directory Layout ---\n")

print("A well-organized directory structure is crucial. Here's a common layout:")
print("""
my_pytorch_project/
├── .git/                   # Git version control
├── .gitignore              # Specifies intentionally untracked files for Git
├── README.md               # Project overview, setup instructions, etc.
├── requirements.txt        # Python dependencies (or environment.yml for Conda)
│
├── configs/                # Configuration files (e.g., YAML, JSON)
│   ├── base_config.yaml
│   └── experiment1_config.yaml
│
├── data/                   # Raw and processed data (often not committed to Git if large)
│   ├── raw/
│   │   └── dataset.csv
│   └── processed/
│       └── train_tokens.pt
│
├── notebooks/              # Jupyter notebooks for exploration, analysis, visualization
│   ├── 01_data_exploration.ipynb
│   └── 02_results_analysis.ipynb
│
├── src/                    # Main source code (or your project_name/)
│   ├── __init__.py
│   ├── data_loader.py      # Dataset classes, data preprocessing
│   ├── model.py            # nn.Module definitions (e.g., your SmallGPT)
│   ├── engine.py           # Training, evaluation, inference loops/functions
│   ├── utils.py            # Utility functions (e.g., saving/loading, metrics)
│   └── CNAME.py            # To specify custom domain for GitHub Pages
│
├── scripts/                # Standalone scripts
│   ├── train.py            # Script to run training
│   ├── evaluate.py         # Script to run evaluation
│   └── generate.py         # Script to generate text
│
├── tests/                  # Unit and integration tests
│   ├── test_data_loader.py
│   └── test_model.py
│
└── checkpoints/            # Saved model checkpoints (usually in .gitignore)
    └── best_model.pth
""")

print("\nKey considerations for this structure:")
print("- Separation of Concerns: Different types of code (data, model, training) are in distinct files/modules.")
print("- `src/` (or `project_name/`): Contains the core Python package for your project.")
print("  This allows you to `import src.model` or `from src.utils import some_func`.")
print("- `scripts/`: Entry points for running tasks. These often import from `src/`.")
print("- `configs/`: Keeps hyperparameters and settings separate from code, making it easy to")
print("  run different experiments without code changes.")
print("- `data/`, `checkpoints/`: Often large, so typically added to `.gitignore` to avoid")
print("  committing them to version control. Store them in cloud storage or a shared drive if needed.")
print("- Version Control (`.git`, `.gitignore`): Essential for tracking changes and collaboration.")


# %% 3. Configuration Management
print("\n--- 3. Configuration Management ---\n")

print("Hardcoding hyperparameters and settings within scripts is bad practice.")
print("Configuration management tools allow you to define these externally.")

# --- Using `argparse` (Standard Python Library) ---
print("`argparse` is good for simple command-line argument parsing.")

# Example: Imagine this is the content of `scripts/train.py`
def example_argparse_setup():
    parser = argparse.ArgumentParser(description="Example Training Script")
    
    # Model arguments
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of Transformer layers')
    
    # Training arguments
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    
    # To simulate running from command line for this example:
    # Replace sys.argv with a list of strings.
    # In a real script, you'd just use: args = parser.parse_args()
    example_args_list = [
        '--embed_dim', '128',
        '--lr', '0.0005',
        '--data_path', '/path/to/your/data', # This will be a dummy path for the example
        '--epochs', '5'
    ]
    # args = parser.parse_args(example_args_list) # Simulate parsing
    
    # If running this cell directly in an interactive environment, parse_args() might
    # try to parse notebook arguments. We'll just show how to access defaults or
    # hypothetical parsed values.
    
    # Accessing arguments (after parsing)
    # print(f"Running with Config:")
    # print(f"  Embedding Dim: {args.embed_dim}")
    # print(f"  Learning Rate: {args.lr}")
    # print(f"  Data Path: {args.data_path}")
    # print(f"  Epochs: {args.epochs}")
    # print(f"  Batch Size (default): {args.batch_size}")
    return parser # Return parser for demonstration purposes

print("Example `argparse` setup (conceptual):")
parser_instance = example_argparse_setup()
# In a real script, you would call:
# args = parser_instance.parse_args()
# print(args)
# To run from command line: python scripts/train.py --lr 0.001 --epochs 20 --data_path ...

print("\n--- Using YAML/JSON files with libraries like Hydra or OmegaConf ---")
print("For more complex configurations, external files (e.g., YAML) are preferred.")
print("Libraries like Hydra (from Facebook Research) or OmegaConf provide powerful ways to:")
print("- Define hierarchical configurations.")
print("- Override settings from the command line.")
print("- Compose configurations from multiple files.")
print("Example `configs/experiment_config.yaml`:")
print("""
model:
  embed_dim: 128
  num_heads: 4
  num_layers: 3
  block_size: 64
  dropout_rate: 0.1

data:
  path: "/datasets/my_text_data"
  tokenizer_name: "gpt2"
  batch_size: 32

training:
  optimizer: "AdamW"
  learning_rate: 3e-4
  num_epochs: 5
  device: "cuda"
  seed: 42
""")
print("You would then load this YAML file in your script (e.g., `train.py`) using a library.")


# %% 4. Code Modularity and Reusability
print("\n--- 4. Code Modularity and Reusability ---\n")

print("Break down your code into logical, reusable components (functions, classes, modules).")
print("Instead of one massive script:")
print("- `src/model.py`: Contains your `nn.Module` definitions (e.g., `SmallGPT`, `TransformerBlock`).")
print("- `src/data_loader.py`: Contains `Dataset` classes and data processing functions.")
print("- `src/engine.py`: Contains functions for training (`train_one_epoch`), evaluation (`evaluate`),")
print("  and inference/generation (`generate_text`). These functions take model, data_loader,")
print("  optimizer, etc., as arguments.")
print("- `src/utils.py`: Holds general helper functions (e.g., saving/loading checkpoints, setting seeds,")
print("  calculating custom metrics).")

print("\nBenefits of modularity:")
print("- Easier to Understand: Smaller, focused pieces of code are simpler to grasp.")
print("- Easier to Test: Individual components can be unit-tested independently.")
print("- Reusability: Model definitions or utility functions can be reused across different scripts")
print("  (e.g., for training, evaluation, and a separate generation script).")
print("- Reduced Redundancy: Avoids copying and pasting code.")


# %% 5. Type Hinting
print("\n--- 5. Type Hinting ---\n")

print("Python is dynamically typed, but type hints (PEP 484) improve code clarity and help catch errors.")
print("Type hints specify the expected types for function arguments, return values, and variables.")
print("They do not enforce types at runtime by default but are used by static analysis tools")
print("like MyPy, Pyright (used by Pylance in VS Code), and by IDEs for better autocompletion and error detection.")

# Example with type hints:
def process_batch(
    batch_data: Dict[str, torch.Tensor], 
    model: nn.Module,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Processes a batch of data, performs a model forward pass, and returns logits and targets.
    
    Args:
        batch_data: A dictionary containing input tensors (e.g., 'input_ids', 'attention_mask').
        model: The PyTorch model.
        device: The device to move data to.
        
    Returns:
        A tuple containing:
            - logits (torch.Tensor): The model's output logits.
            - targets (torch.Tensor): The target tensor from the batch.
    """
    # This is a conceptual example; actual content depends on your batch structure
    input_ids = batch_data['input_ids'].to(device)
    targets = batch_data.get('labels', input_ids.clone()).to(device) # Example: use input_ids as labels if not present
    
    logits = model(input_ids)
    return logits, targets

# Another example
def get_sequence_lengths(sequences: List[List[int]], default_len: Optional[int] = None) -> List[int]:
    if not sequences and default_len is not None:
        return [default_len]
    return [len(seq) for seq in sequences]

print("\nUsing `typing` module for more complex types: `List`, `Dict`, `Tuple`, `Optional`, `Callable`, etc.")
print("Type hints make function signatures much clearer about expected inputs and outputs.")


# %% 6. Logging
print("\n--- 6. Logging ---\n")

print("The `logging` module is more robust and flexible than `print()` statements for tracking events,")
print("debugging, and monitoring training progress, especially for long-running processes.")

# Basic logging configuration:
# This setup is often done once at the beginning of your main script (e.g., train.py).
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format) # Set default level to INFO

# Get a logger instance (typically one per module)
logger = logging.getLogger(__name__) # Uses the current module's name

# Example usage:
logger.debug("This is a debug message. (Will not show if level is INFO)")
logger.info("Starting training process...")
learning_rate_example = 0.001
logger.info(f"Using learning rate: {learning_rate_example}")

try:
    # Simulate an operation that might fail
    # x = 1 / 0 # Uncomment to see an error logged
    logger.info("Some operation completed successfully.")
except Exception as e:
    logger.error(f"An error occurred: {e}", exc_info=True) # exc_info=True logs traceback

logger.warning("Validation accuracy is low. Consider checking model or data.")
logger.critical("Critical error - system shutting down (hypothetical).")

print("\nBenefits of `logging`:")
print("- Log Levels: Control verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL).")
print("- Output Destinations: Log to console, files, or even network services.")
print("- Formatting: Customizable log message format (timestamp, level, module name).")
print("- Centralized Control: Configure logging behavior for the entire application.")
print("Consider using different loggers for different parts of your application, e.g.,")
print("`data_logger = logging.getLogger('src.data_loader')`")


# %% 7. Unit Testing (Conceptual Overview)
print("\n--- 7. Unit Testing (Conceptual Overview) ---\n")

print("Unit tests verify that individual components (functions, classes) of your code work correctly.")
print("For ML projects, this could include testing:")
print("- Data processing functions: Do they handle edge cases? Produce correct shapes/types?")
print("- `Dataset` implementations: Does `__getitem__` return expected data? Does `__len__` work?")
print("- Model components (layers): Do they produce correct output shapes given known input shapes?")
print("  (Testing exact numerical outputs can be hard due to random initializations, but shape is testable).")
print("- Utility functions: Do they perform their calculations correctly?")

print("\nTools: Python's `unittest` module, or more popular alternatives like `pytest`.")
print("Example `tests/test_data_loader.py` (conceptual, using pytest style):")
print("""
import torch
from src.data_loader import MyCustomDataset # Assuming this exists

def test_my_custom_dataset_len():
    raw_data = [{"text": "hello"}, {"text": "world"}]
    dataset = MyCustomDataset(raw_data, tokenizer=...) # tokenizer would be a mock or simple one
    assert len(dataset) == 2

def test_my_custom_dataset_getitem():
    raw_data = [{"text": "test"}]
    dataset = MyCustomDataset(raw_data, tokenizer=...)
    sample_input_ids, sample_target_ids = dataset[0]
    assert isinstance(sample_input_ids, torch.Tensor)
    assert sample_input_ids.dtype == torch.long
    # ... more assertions about shape, content if deterministic
""")

print("\nBenefits of Unit Testing:")
print("- Early Bug Detection: Catch issues before they propagate.")
print("- Code Confidence: Provides assurance when refactoring or adding features.")
print("- Documentation: Tests serve as examples of how components are meant to be used.")


# %% 8. Dependency Management
print("\n--- 8. Dependency Management ---\n")

print("Clearly define and manage your project's Python dependencies.")
print("This ensures that anyone else (or your future self) can set up the correct environment.")

print("- `requirements.txt` (pip):")
print("  A plain text file listing packages and optionally their versions.")
print("  Example content:")
print("""
torch>=1.10.0,<2.0.0
torchvision
transformers==4.20.0
numpy
tqdm
  """)
print("  Generated via: `pip freeze > requirements.txt` (captures all packages in env).")
print("  Better to curate it manually or use tools like `pip-tools` for cleaner dependencies.")
print("  Installed via: `pip install -r requirements.txt`")

print("\n- Conda Environment (`environment.yml`):")
print("  Used with Anaconda/Miniconda. Can manage Python versions and non-Python packages too.")
print("  Example content:")
print("""
name: my_llm_env
channels:
  - pytorch
  - defaults
  - conda-forge
dependencies:
  - python=3.9
  - pytorch::pytorch
  - torchvision
  - cudatoolkit=11.3 # Example for specific CUDA version with PyTorch channel
  - transformers
  - numpy
  - pip:
    - hydra-core
    - tqdm
""")
print("  Created via: `conda env create -f environment.yml`")
print("  Updated via: `conda env update -f environment.yml --prune`")

print("\nUsing Virtual Environments (e.g., `venv`, `conda`) is highly recommended to isolate project dependencies.")


# %% Conclusion
print("\n--- Module 10 Summary ---\n")
print("Key Takeaways:")
print("- SWE practices are vital for robust, maintainable, and reproducible ML projects.")
print("- A logical directory structure separates concerns (data, source code, scripts, configs, tests).")
print("- Externalize configurations using `argparse` for simple cases or YAML/Hydra for complex ones.")
print("- Write modular code: break down tasks into reusable functions and classes in separate files.")
print("- Use type hints to improve code clarity and enable static analysis.")
print("- Employ the `logging` module for better tracking and debugging than `print()`.")
print("- Implement unit tests (e.g., with `pytest`) to verify individual components.")
print("- Manage dependencies explicitly using `requirements.txt` or `environment.yml` within virtual environments.")
print("\nAdopting these practices will significantly improve the quality and efficiency of your PyTorch projects,")
print("especially as they grow in complexity or involve collaboration.")

print("\nEnd of Module 10.")