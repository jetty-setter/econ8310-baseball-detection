"""
setup.py — Run this ONCE before train.py to install all dependencies.

Open Command Prompt and run:
    python setup.py
"""

import subprocess
import sys

packages = [
    "ultralytics",       # YOLOv8 — includes PyTorch
    "opencv-python",     # image processing
    "matplotlib",        # plots
    "PyYAML",            # YAML config files
]

print("Installing required packages...")
print("This may take a few minutes the first time.\n")

for pkg in packages:
    print(f"Installing {pkg}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

print("\n✓ All packages installed successfully!")
print("\nNext step: run the training pipeline:")
print("    python train.py")
