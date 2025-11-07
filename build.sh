#!/bin/bash

echo "====================================="
echo "Driver Monitoring System Build Script"
echo "====================================="

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install build dependencies first
echo "Installing build dependencies..."
pip install --no-cache-dir setuptools wheel Cython cmake

# Install numpy first (required by others)
echo "Installing numpy..."
pip install --no-cache-dir "numpy>=1.26.0,<2.0.0"

# Install OpenCV
echo "Installing OpenCV..."
pip install --no-cache-dir opencv-python-headless

# Install dlib (this takes the longest)
echo "Installing dlib..."
pip install --no-cache-dir dlib

# Install remaining packages
echo "Installing Flask and other dependencies..."
pip install --no-cache-dir -r requirements.txt

echo "====================================="
echo "Build completed successfully!"
echo "====================================="