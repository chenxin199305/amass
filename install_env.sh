#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# Update the package list
echo "Updating package list..."
sudo apt-get update -y
echo ""

# Upgrade pip to the latest version
echo "Upgrading pip to the latest version..."
pip install --upgrade pip
echo ""

# Install the required packages
echo "Installing required packages..."
pip install -e . -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
echo ""

# Success message
echo "All packages installed and datasets downloaded successfully."