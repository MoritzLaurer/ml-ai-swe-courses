#!/bin/bash

# Module A0: Setting Up Your Development Environment (with uv)
# This script guides you through setting up your environment for the course.
# The actual commands should be run in your terminal.

# --- Introduction to uv ---
# uv is a very fast Python package installer and resolver, written in Rust.
# It's designed to be a drop-in replacement for pip and pip-tools.
# We'll use it to manage our project's virtual environment and dependencies.
# For more info (as of 2025), refer to the official uv documentation: https://github.com/astral-sh/uv

# --- 1. Installing uv ---
# Follow the official installation instructions for your operating system.
# For macOS and Linux, a common method is:
# curl -LsSf https://astral.sh/uv/install.sh | sh
# Verify installation by running:
# uv --version
# (Ensure the version is recent, reflecting 2025 standards)

# --- 2. Creating a Project Directory ---
# If you haven't already, create a main directory for this course.
# For example:
# mkdir databases
# cd databases

# --- 3. Initializing a Virtual Environment with uv ---
# Inside your project directory, create a virtual environment:
# uv init
# uv venv

# --- 4. Activating the Virtual Environment ---
# Activate the environment. The command depends on your shell:
# For bash/zsh:
# source .venv/bin/activate
# For fish:
# source .venv/bin/activate.fish
# For PowerShell:
# .venv\Scripts\Activate.ps1
# Your terminal prompt should change to indicate the active environment.

# --- 5. Installing Core Packages and Initializing Project Dependencies ---
# For Mac Zsh terminal users:
# uv add "psycopg[binary]" python-dotenv
#
# For other shells (like Bash):
# uv add psycopg[binary] python-dotenv
#
# After running this, you should see a `pyproject.toml` and potentially a `uv.lock`
# file in your project directory. It's good practice to commit both of these
# files to your version control system (e.g., Git).
# We will add other dependencies in the respective modules.
#
# Note on psycopg:
# - 'psycopg' should refer to psycopg3 by 2025.
# - The [binary] extra installs a pre-compiled version, which is often easier.
# - If you encounter issues or prefer to compile from source, you might use 'psycopg'
#   and ensure you have the necessary build tools (like libpq-dev).

# --- 6. Setting up PostgreSQL ---
# You need a PostgreSQL database (latest stable version recommended).
# Options:
#
#   a) Local Installation:
#      - macOS: `brew install postgresql`
#      - Linux (Debian/Ubuntu): `sudo apt update && sudo apt install postgresql postgresql-contrib`
#      - Windows: Download the installer from the official PostgreSQL website.
#      - After installation, ensure the PostgreSQL server is running.
#      - You'll also need to create a database user and a database for this project.
#        Example psql commands (run as postgres superuser or a user with CREATEDB rights):
#        CREATE USER myprojectuser WITH PASSWORD 'yoursecurepassword';
#        CREATE DATABASE myprojectdb OWNER myprojectuser;
#
#   b) Docker (Recommended for ease of use and version consistency):
#      - Ensure you have Docker installed and running.
#      - Create a `docker-compose.yml` file in your project directory (example in the repo)
#      - Run `docker compose up -d` in the directory with the `docker-compose.yml` file.
#      - To connect from your host machine, the host will be 'localhost' and port '5432'.

# --- 7. Creating a .env File ---
# Create a file named `.env` in your project root directory.
# This file will store your database connection details, and `python-dotenv`
# will load them as environment variables.
#
# Add the following to your .env file, replacing with your actual details:
#
# DB_HOST=localhost
# DB_PORT=5432 
# DB_NAME=myprojectdb
# DB_USER=myprojectuser
# DB_PASSWORD=yoursecurepassword
#
# **IMPORTANT: Add `.env` to your `.gitignore` file to avoid committing secrets!**
# echo ".env" >> .gitignore
# echo ".venv/" >> .gitignore
# echo "__pycache__/" >> .gitignore
# echo "*.pyc" >> .gitignore

# --- 8. Verifying the Setup ---
# Create a small Python script (e.g., `verify_setup.py`) in your project directory
# to test the database connection.
#
# Run this script from your activated virtual environment:
# python verify_setup.py
#
# If it prints the PostgreSQL version, your basic setup is complete!

echo "Setup script 'A0_setup.sh' created. Please open it and follow the instructions."
echo "Remember to execute the commands in your terminal, not by running this script directly (unless you modify it to do so)." 