#!/bin/bash

# Script to run the PyTorch training script in the background using nohup.

# --- Configuration ---
PROJECT_DIR="/Users/moritzlaurer/Library/CloudStorage/Dropbox/hugging_face/huggingface/projects/swe-learning/pytorch"
VENV_PYTHON_PATH="${PROJECT_DIR}/.venv/bin/python"
SCRIPT_TO_RUN="08_full_gpt_model_train.py"
OUTPUT_LOG_FILE="${PROJECT_DIR}/checkpoints/logs/08_training_output_$(date +%Y%m%d_%H%M%S).log" # Unique log file per run

# --- Main execution ---

# 1. Navigate to the project directory
echo "Changing directory to ${PROJECT_DIR}..."
cd "${PROJECT_DIR}" || { echo "Failed to change directory to ${PROJECT_DIR}. Exiting."; exit 1; }

# 2. Check if virtual environment Python exists
if [ ! -f "${VENV_PYTHON_PATH}" ]; then
    echo "Error: Python interpreter not found at ${VENV_PYTHON_PATH}."
    echo "Please ensure your virtual environment is set up correctly in .venv/"
    exit 1
fi

# 3. Check if the script to run exists
if [ ! -f "${SCRIPT_TO_RUN}" ]; then
    echo "Error: Training script ${SCRIPT_TO_RUN} not found in ${PROJECT_DIR}."
    exit 1
fi

echo "Starting training script: ${SCRIPT_TO_RUN}"
echo "Output will be logged to: ${OUTPUT_LOG_FILE}"

# 4. Run the script with nohup
nohup "${VENV_PYTHON_PATH}" "${SCRIPT_TO_RUN}" > "${OUTPUT_LOG_FILE}" 2>&1 &

# Get the PID of the backgrounded process
PID=$!

echo "Training started in the background with PID: ${PID}."
echo "You can monitor the output with: tail -f ${OUTPUT_LOG_FILE}"
echo "To check if it's running: ps -p ${PID}"
echo "To stop the script, use: kill ${PID}"

exit 0