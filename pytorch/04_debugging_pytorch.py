# -*- coding: utf-8 -*-
# 04_debugging_pytorch.py

# Module 4: Debugging PyTorch Code
#
# This script covers common debugging techniques in PyTorch:
# 1. Leveraging Eager Execution with `print()` statements.
# 2. Using Python Debuggers (`pdb`, IDE debuggers via `breakpoint()`).
# 3. Diagnosing Shape Mismatches.
# 4. Investigating Gradient Issues (`None` gradients, checking norms).
# 5. Catching Device Mismatches.
# 6. Comparisons to debugging JAX.

import torch
import torch.nn as nn
import torch.optim as optim

print("--- Module 4: Debugging PyTorch Code ---\n")
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


# %% Setup: A Simple Model and Training Step Components
print("\n--- Setup: Simple Model and Training Step Components ---\n")

class DebugModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        # Potential place for errors later
        self.factor = nn.Parameter(torch.ones(1) * 5.0)

    def forward(self, x, debug_prints=False):
        if debug_prints: 
            print(f"Input shape: {x.shape}, dtype: {x.dtype}, device: {x.device}")

        x = self.layer1(x)
        if debug_prints: 
            print(f"After layer1 shape: {x.shape}")

        x = self.relu(x)
        if debug_prints: 
            print(f"After relu shape: {x.shape}")

        # Let's introduce a potential manipulation before layer2
        # Maybe a reshape or matmul could go wrong here in a real scenario
        # x = x.view(x.shape[0], -1) # Example manipulation

        x = self.layer2(x)
        if debug_prints: 
            print(f"After layer2 shape: {x.shape}")

        # Multiply by a learnable scalar parameter
        x = x * self.factor
        if debug_prints: 
            print(f"After factor mult shape: {x.shape}")

        return x

# Instantiate model, loss, optimizer
input_d, hidden_d, output_d = 8, 16, 3
model = DebugModel(input_d, hidden_d, output_d).to(device)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy data
batch_size = 4
dummy_input = torch.randn(batch_size, input_d, device=device)
dummy_target = torch.randn(batch_size, output_d, device=device)

print("Setup complete. Model, loss, optimizer, and dummy data created.")


# %% 1. Leveraging Eager Execution & Print Statements
print("\n--- 1. Leveraging Eager Execution & Print Statements ---\n")

# PyTorch runs operations immediately (eagerly), just like standard Python.
# This means `print()` is your best friend for checking tensor properties anywhere.
print("Running forward pass with debug_prints=True:")
# (Ensure model is on the correct device from setup)
model.train() # Set to train mode
with torch.no_grad(): # Use no_grad if just checking forward pass without training
    output = model(dummy_input, debug_prints=True)

print("\nPrint statements inside the forward pass show intermediate shapes.")
print("This is invaluable for tracing where dimensions change unexpectedly.")


# %% 2. Using Python Debuggers (`pdb`, `breakpoint()`)
print("\n--- 2. Using Python Debuggers (`pdb`, `breakpoint()`) ---\n")

# Standard Python debuggers work perfectly with PyTorch code.
# - `import pdb; pdb.set_trace()`: Classic way to set a breakpoint.
# - `breakpoint()`: Newer (Python 3.7+) and cleaner way.

# Example: Place `breakpoint()` inside the forward method or training loop.
# (We won't run this automatically, but showing where to put it)
print("Imagine placing `breakpoint()` inside the forward method:")
print("""
    def forward(self, x, debug_prints=False):
        ...
        x = self.layer1(x)
        print("About to enter debugger...")
        breakpoint() # Execution will pause here
        print("Resuming after debugger...")
        x = self.relu(x)
        ...
        return x
""")
print("When execution pauses, you can use commands in the terminal:")
print("  'p variable_name' (print variable value/shape)")
print("  'n' (next line)")
print("  'c' (continue execution)")
print("  'q' (quit debugger)")
print("This allows interactive inspection of tensors and program state.")


# %% 3. Debugging Shape Mismatches
print("\n--- 3. Debugging Shape Mismatches ---\n")

# Shape errors are very common (e.g., matrix multiply requires specific dims).
# PyTorch usually gives informative RuntimeErrors.

# --- Example Error ---
# Let's simulate passing input with the wrong input dimension to the model
wrong_input_dim = input_d + 1 # 9 instead of 8
wrong_input = torch.randn(batch_size, wrong_input_dim, device=device)

print("Attempting forward pass with wrong input shape:")
try:
    model.eval()
    with torch.no_grad():
        output = model(wrong_input)
except RuntimeError as e:
    print(f"Caught expected RuntimeError:\n{e}")
    print("\nAnalysis:")
    print("The error message clearly states the mismatched shapes for mat1 and mat2")
    print("in the `nn.Linear` layer (layer1 in our model).")
    print("It expected input features =", input_d, "but got", wrong_input_dim)
    print("Debug Strategy: Add print(x.shape) or use debugger right before the failing layer (layer1).")

model.train() # Reset model state


# %% 4. Debugging Gradient Issues
print("\n--- 4. Debugging Gradient Issues ---\n")

# --- None Gradients ---
# Gradients (`.grad`) might be `None` if a parameter wasn't used in the loss calculation,
# if `requires_grad=False`, or if operations were done in `torch.no_grad()`.

# Example: Parameter not involved in loss
unused_param = nn.Parameter(torch.randn(5, requires_grad=True)).to(device)
# Our main model calculation:
optimizer.zero_grad()
output = model(dummy_input)
loss = loss_fn(output, dummy_target)
loss.backward() # Calculate gradients for model parameters

print("Checking gradients after backward pass:")
print(f"Gradient for model.layer1.weight (exists): {model.layer1.weight.grad is not None}")
# Try accessing grad of the unused parameter
print(f"Gradient for unused_param (should be None): {unused_param.grad}")
optimizer.step() # This would only update model parameters, not unused_param

# --- Checking Gradient Values (Exploding/Vanishing) ---
# Sometimes gradients become too large (explode) or too small (vanish).
print("\nChecking gradient norms (a way to spot exploding/vanishing gradients):")
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2) # Calculate L2 norm of gradients
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5
print(f"  L2 Norm of all model gradients: {total_norm:.4f}")

# If the norm is huge (NaN or Inf), gradients might be exploding.
# Solution: Gradient Clipping (often done before optimizer.step())
# `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`

# If the norm is tiny, gradients might be vanishing.
# Solutions: Better initialization, different activation functions (ReLU helps), residual connections, normalization layers.

# Using Hooks for detailed grad inspection (Advanced)
# Hooks are functions called when gradients are computed.
# handle = model.layer1.weight.register_hook(lambda grad: print(f"  Hook sees grad shape for layer1.weight: {grad.shape}"))
# loss.backward(retain_graph=True) # Need retain_graph if calling backward multiple times or using hooks extensively
# handle.remove() # Important to remove hooks when done


# %% 5. Debugging Device Mismatches
print("\n--- 5. Debugging Device Mismatches ---\n")

# Operations require all involved tensors to be on the same device.

# --- Example Error ---
# Create a tensor explicitly on CPU
cpu_tensor = torch.randn(batch_size, output_d, device='cpu')

print("Attempting loss calculation with target tensor on CPU:")
try:
    # 'output' is on the 'device' (e.g., GPU/MPS) determined earlier
    # 'cpu_tensor' is on CPU
    loss = loss_fn(output, cpu_tensor) # output is on 'device', cpu_tensor is on 'cpu'
except RuntimeError as e:
    print(f"Caught expected RuntimeError:\n{e}")
    print("\nAnalysis:")
    print("Error clearly states tensors are expected on the same device but found different ones.")
    print("Debug Strategy: Check `.device` attribute for all inputs to an operation.")
    print(f"  output.device: {output.device}")
    print(f"  cpu_tensor.device: {cpu_tensor.device}")
    print("Solution: Move tensors to the same device using `.to(device)`.")
    # Correct way:
    loss = loss_fn(output, cpu_tensor.to(output.device))
    print(f"Loss calculated successfully after moving target to {output.device}: {loss.item():.4f}")


# %% 6. JAX Comparison
print("\n--- 6. JAX Comparison ---\n")

print("Debugging `jit`-compiled JAX code can be different:")
print("- Standard `print()` doesn't work reliably inside `@jax.jit` functions (runs only during trace).")
print("- Use `jax.debug.print(\"Value: {x}\", x)` or `jax.debug.callback(my_python_func, arg)` for visibility inside JIT.")
print("- Python debuggers (`pdb`, `breakpoint()`) don't step into compiled code.")
print("- Strategies include:")
print("  - Running code eagerly (without `@jit`) for debugging.")
print("  - Using `jax.disable_jit()` context manager.")
print("  - Analyzing the JAX traceback carefully (can sometimes be complex).")
print("- Shape errors often occur during the initial JIT compilation/trace, not necessarily at runtime like PyTorch.")
print("- Device mismatches are handled differently; JAX often tries to stage data automatically but explicit control is via `jax.device_put`.")
print("\nPyTorch's eager execution generally offers a more conventional debugging experience.")


# %% Conclusion
print("\n--- Module 4 Summary ---\n")
print("Key Takeaways:")
print("- Use `print()` extensively in PyTorch for shapes, dtypes, devices.")
print("- Standard Python debuggers (`pdb`, `breakpoint()`) work seamlessly.")
print("- Read `RuntimeError` messages carefully for shape and device mismatches.")
print("- Check `.grad` attributes; `None` indicates a disconnected graph or no `requires_grad`.")
print("- Monitor gradient norms to spot exploding/vanishing issues.")
print("- Ensure all tensors in an operation are on the same device.")
print("- PyTorch debugging often feels more direct than debugging `jit`-compiled JAX code.")

print("\nEnd of Module 4.")