# C6_debugging_profiling.py

# Welcome to Advanced Module C.6: Debugging & Profiling JAX Code!
# Objective: Learn techniques to debug JAX code within `jit` and `pmap`, and
#            profile code execution to identify performance bottlenecks.
# Theme Integration: We'll use simple functions and conceptually apply these
#                    techniques to the training steps developed earlier.

import jax
import jax.numpy as jnp
import jax.random as random
import time
import os



# Configuration for demonstration
# Optional: Force CPU to ensure consistent behavior for some debug tools if GPU/TPU varies
# jax.config.update('jax_platform_name', 'cpu')

print("--- C6.1 Debugging Challenges in JAX ---")
# - `jit`: Python code runs during *tracing* (once per input shape/type), not necessarily
#   during each execution of the compiled code. Standard `print`/`pdb` are unreliable inside.
# - `pmap`: Code runs in parallel on multiple devices. Debugging needs to consider this.
# - Async Dispatch: JAX operations return control quickly; execution happens later on the
#   device. Errors might appear detached from the source line. `block_until_ready()` helps.

def basic_computation(x):
  y = jnp.sin(x) * 2.0
  z = jnp.log(y + 1.0)
  return z

# Force sync execution for predictable timing/error location *during debugging*
output = basic_computation(jnp.array(1.0))
output.block_until_ready() # Waits for the computation graph above to finish
print(f"Basic computation output: {output:.4f}")


print("\n" + "="*30 + "\n")


# --- C6.2 Debugging Runtime Values ---
print("--- C6.2 Debugging Runtime Values ---")

# --- Using `jax.debug.print` ---
# Prints values from compiled/parallel code by embedding prints into the XLA graph.
@jax.jit
def computation_with_debug_print(x):
  # Use {} formatting and pass keyword args
  jax.debug.print("Input x: {x}", x=x)
  y = jnp.sin(x) * 2.0
  # Can print intermediate values, shapes, dtypes etc.
  jax.debug.print("Intermediate y: {y}, shape={s}, dtype={d}", y=y, s=y.shape, d=y.dtype)
  z = jnp.log(y + 1.0)
  jax.debug.print("Output z: {z}", z=z)
  return z

print("Running jitted function with jax.debug.print:")
result = computation_with_debug_print(jnp.array(2.0))
result.block_until_ready() # Ensure prints flush before continuing
print(f"Final result (debug print): {result:.4f}")
# Note: The prints occur during device execution. Order relative to host prints may vary without blocking.

# --- Using `jax.debug.breakpoint` ---
# Pauses execution *at runtime* inside compiled code and enters a pdb-like shell.
# NOTE: This requires an interactive terminal session. Running in some non-interactive
#       environments might cause hangs or errors.
@jax.jit
def computation_with_breakpoint(x):
  y = jnp.sin(x) * 2.0
  print(f"(Python print during trace): Input x = {x}") # This runs only during trace time
  jax.debug.print("Value y before breakpoint: {y}", y=y)
  jax.debug.breakpoint() # Execution pauses here at runtime
  # In the breakpoint() shell, you can inspect `x`, `y` etc. Type 'c' to continue.
  z = jnp.log(y + 1.0)
  return z

print("\nRunning jitted function with jax.debug.breakpoint:")
print("-> Execution will pause. Type 'c' and press Enter in the terminal prompt to continue.")
try:
    result_bp = computation_with_breakpoint(jnp.array(3.0))
    result_bp.block_until_ready()
    print(f"Final result (breakpoint): {result_bp:.4f}")
except Exception as e:
    print(f"\nError running breakpoint (possibly non-interactive environment): {e}")
print("-> Breakpoint section finished.")
# NOTE: Uncomment the section above to try `jax.debug.breakpoint` interactively.


# --- Using `io_callback` (External Callback) ---
# Sends data back to host to run arbitrary Python code (e.g., pdb, complex logging).
# Breaks purity, adds overhead. Use with caution. Replaces deprecated host_callback.
from jax.experimental import io_callback # Use io_callback instead of host_callback

def host_pdb_callback(arg): # Removed transform_params
  # This function runs on the Python host process
  print("\n--- Entered host callback ---")
  print(f"Received arg with shape {arg.shape}, dtype {arg.dtype}")
  # Can inspect the value, save it, or drop into debugger
  import pdb; pdb.set_trace()
  print("--- Exiting host callback ---")

@jax.jit
def computation_with_io_callback(x):
    y = jnp.sin(x) * 2.0
    # io_callback calls the host function with 'y'. Result shape is None as it doesn't return to JAX.
    io_callback(host_pdb_callback, None, y) # Pass value(s) to host
    y_tapped = y # Simulate the pass-through behavior of id_tap
    # Can continue computation with y_tapped (which equals y)
    z = jnp.log(y_tapped + 1.0)
    return z

print("\nRunning jitted function with io_callback:")
print("-> Execution will pause in host PDB. Type 'c' and press Enter to continue.")
try:
    result_ioc = computation_with_io_callback(jnp.array([4.0, 5.0]))
    result_ioc.block_until_ready()
    print(f"Final result (io_callback): {result_ioc}")
except Exception as e:
    print(f"\nError running io_callback: {e}")
print("-> IO callback section finished.")
# NOTE: Uncomment the section above to try `io_callback`.

print("\n" + "="*30 + "\n")



# --- C6.3 Debugging Errors ---
print("--- C6.3 Debugging Errors ---")

# --- Disabling JIT ---
# Easiest way to debug logic errors: turn off JIT temporarily.
print("\nDisabling JIT temporarily:")
try:
    with jax.disable_jit():
        # Now standard print and pdb work as expected inside the function
        def func_with_error(x):
            y = x - 1.0
            y = jnp.where(y == 0, 0.0, 1.0 / y) # Potential division by zero
            print(f"(Inside disable_jit) Intermediate y: {y}")
            # import pdb; pdb.set_trace() # Can use pdb here
            return jnp.sum(y)

        result_no_jit = func_with_error(jnp.array([0.0, 1.0, 2.0]))
        print(f"Result with disable_jit: {result_no_jit}") # Might produce inf
except Exception as e:
    print(f"Error occurred with JIT disabled: {e}")

print("JIT is automatically re-enabled outside the 'with' block.")


# --- NaN Debugging ---
# Automatically detect NaNs and pinpoint the source operation.
print("\nEnabling NaN debugging:")
# jax.config.update("jax_debug_nans", True) # Enable globally

@jax.jit
def cause_nan(x):
    # Example: log of negative number
    return jnp.log(x - 5.0)

try:
    # Wrap the potentially problematic code in a context manager for NaN checking
    # or enable globally with config.update above.
    with jax.debug_nans(True):
      print("Running function that might produce NaN...")
      result_nan = cause_nan(jnp.array(4.0))
      result_nan.block_until_ready()
      print(f"Result (NaN check): {result_nan}") # Should not reach here if NaN occurs
except FloatingPointError as e:
    print(f"SUCCESS: Caught FloatingPointError (likely NaN): {e}")
except Exception as e:
    print(f"Caught other error during NaN check: {e}")

# Disable NaN checking if enabled globally
# jax.config.update("jax_debug_nans", False)
print("Finished NaN check section.")


# --- Understanding Tracebacks ---
# JAX tracebacks can be long due to nested function transformations.
# Tips:
# - Look for your file names and line numbers.
# - Identify the specific primitive operation that failed (e.g., `dot_general`, `reduce_sum`).
# - The error message often indicates the type of failure (shape mismatch, dtype error, NaN).
# - Use the debugging tools above to inspect values leading up to the error.

print("\n" + "="*30 + "\n")



# --- C6.4 Profiling JAX Code ---
print("--- C6.4 Profiling JAX Code ---")
# Profiling helps find performance bottlenecks (CPU vs GPU/TPU time, memory vs compute).

# --- Using `jax.profiler` for Detailed Traces ---
# Captures execution traces viewable in TensorBoard.

profile_log_dir = "./tensorboard_logs" # Directory to save trace files
print(f"Preparing to profile. Traces will be saved to: {profile_log_dir}")
os.makedirs(profile_log_dir, exist_ok=True)

# Simple function to profile (use something slightly more substantial than before)
@jax.jit
def moderately_complex_computation(x, w1, w2):
    y = jnp.dot(jnp.sin(x), w1)
    z = jnp.dot(jnp.tanh(y), w2)
    return z.mean()

key, k1, k2, k3 = random.split(random.PRNGKey(202), 4)
data = random.normal(k1, (500, 100))
w1 = random.normal(k2, (100, 200))
w2 = random.normal(k3, (200, 50))

# --- Method 1: Start/Stop ---
# print("\nProfiling using start/stop trace...")
# jax.profiler.start_trace(profile_log_dir)
#
# # Run the code section you want to profile (e.g., a few training steps)
# for _ in range(5): # Profile 5 iterations
#     result_prof = moderately_complex_computation(data, w1, w2)
#     result_prof.block_until_ready() # Ensure computation finishes within trace
#
# jax.profiler.stop_trace()
# print(f"Trace saved to {profile_log_dir}. View with TensorBoard.")
# print("Run: tensorboard --logdir ./tensorboard_logs")
# NOTE: Uncomment above section to generate a profile trace.

# --- Method 2: Context Manager ---
print("\nProfiling using context manager...")
try:
    with jax.profiler.trace(profile_log_dir, create_perfetto_link=False):
        # Run the code section to profile
        for i in range(5):
            result_prof = moderately_complex_computation(data, w1, w2)
            result_prof.block_until_ready()
            print(f"   profiled iter {i+1}, result: {result_prof:.4f}")
    print(f"Trace saved to {profile_log_dir}. View with TensorBoard.")
    print("Run: tensorboard --logdir ./tensorboard_logs")
except Exception as e:
    print(f"Could not profile (ensure dependencies are installed): {e}")


# --- Viewing Profiles ---
# - Install TensorBoard: `pip install tensorboard`
# - Run from terminal: `tensorboard --logdir ./tensorboard_logs`
# - Open the URL provided in your browser.
# - Navigate to the "Profile" tab.
# - Look for: device utilization, kernel execution times, memory operations, host/device interactions.

# --- Simple Timing (with blocking) ---
# For quick checks, but less informative than profiling.
print("\nSimple timing using time.time() with block_until_ready():")
start_time = time.time()
result_timed = moderately_complex_computation(data, w1, w2)
result_timed.block_until_ready() # CRITICAL for accurate timing
end_time = time.time()
print(f"Synchronous execution time: {end_time - start_time:.6f} seconds")

print("\n" + "="*30 + "\n")


# --- C6 Summary ---
# - Debugging JIT/pmap code often requires JAX-specific tools due to tracing & parallelism.
# - Use `jax.debug.print("{x}", x=value)` to print values from compiled device code.
# - Use `jax.debug.breakpoint()` for interactive debugging within compiled code (requires interactive terminal).
# - Use `jax.experimental.host_callback.id_tap` to call arbitrary Python host code (like `pdb`) from device
#   (Note: `host_callback` is deprecated, prefer `jax.experimental.io_callback` or `jax.debug.callback`).
# - Temporarily disable JIT using `with jax.disable_jit():` for standard Python debugging.
# - Enable NaN checks globally (`jax.config.update("jax_debug_nans", True)`) or locally (`with jax.debug_nans(True):`)
#   to automatically raise errors on NaN/inf.
# - Use `jax.profiler` (start/stop trace or context manager) to capture detailed performance traces
#   viewable in TensorBoard (`tensorboard --logdir ./your_log_dir`).