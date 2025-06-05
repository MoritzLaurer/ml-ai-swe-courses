#!/bin/bash

# Bash script to set up the initial TypeScript project structure and dependencies
# for the 'typescript' curriculum.
# Note: This script assumes it's run in a directory where you want to create 'typescript'.

# --- Configuration ---
PROJECT_DIR="typescript"
FIRST_MODULE_DIR="00-setup"
TS_CONFIG_FILE="tsconfig.json"
HELLO_TS_FILE="${FIRST_MODULE_DIR}/hello-typescript.ts"

# --- Commands ---

echo "--- Setting up project directory: ${PROJECT_DIR} ---"
# Create the main project directory
mkdir "${PROJECT_DIR}"
# Change into the project directory
cd "${PROJECT_DIR}" || exit # Exit if cd fails

echo "--- Initializing npm project ---"
# Create a package.json file with default settings
npm init -y

echo "--- Installing development dependencies ---"
# Install TypeScript, ts-node (for running TS directly), tsx (for running TS directly avoiding import/export issues), and Node.js types
# Running `npm install` for the first time in the dir with package.json (from npm init -y)
# this creates the node_modules folder, similar to venv in Python
# Also creates package-lock.json for exact module versions
npm install typescript ts-node tsx @types/node --save-dev

echo "--- Creating tsconfig.json ---"
# Create the TypeScript configuration file
tsc --init
# Should then be manually updated

echo "--- Creating first module directory and file ---"
# Create the directory for the first module
mkdir "${FIRST_MODULE_DIR}"

# Create the initial hello-typescript.ts file
touch "${HELLO_TS_FILE}"
# Manually add the content to the file

echo "--- Setup Complete! ---"
echo "You can now run the first script using:"
echo "npx ts-node ${HELLO_TS_FILE}"
echo "or"
echo "npx tsx ${HELLO_TS_FILE}"
echo "-----------------------"

# Difference between ts-node and tsx:
# ts-node: 
# + transpiles in memory without creating separate .js files. 
# + does type checks during transpilation
# - causes issues when using import/export
# tsx (commonly used today):
# + also transpiles in memory without creating separate .js files. 
# - does not do type checks during transpilation. 
#   But type checking can also be handled by IDE or separate type-check tool.
# + is faster than ts-node
# + aims to "just work". e.g. avoids issues with import/export
