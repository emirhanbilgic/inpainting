#!/usr/bin/env python3
"""
Environment diagnostic script.
Run this on your server to check Python version and package versions.
"""
import sys
import platform

print("=" * 60)
print("ENVIRONMENT DIAGNOSTICS")
print("=" * 60)

print(f"\nPython version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Platform: {platform.platform()}")

print("\n--- Package Versions ---")
try:
    import torch
    print(f"torch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("torch: NOT INSTALLED")

try:
    import open_clip
    print(f"open_clip: {open_clip.__version__}")
except ImportError:
    print("open_clip: NOT INSTALLED")

try:
    import numpy as np
    print(f"numpy: {np.__version__}")
except ImportError:
    print("numpy: NOT INSTALLED")

try:
    import torchvision
    print(f"torchvision: {torchvision.__version__}")
except ImportError:
    print("torchvision: NOT INSTALLED")

try:
    import PIL
    print(f"Pillow: {PIL.__version__}")
except ImportError:
    print("Pillow: NOT INSTALLED")

try:
    import h5py
    print(f"h5py: {h5py.__version__}")
except ImportError:
    print("h5py: NOT INSTALLED")

try:
    import sklearn
    print(f"scikit-learn: {sklearn.__version__}")
except ImportError:
    print("scikit-learn: NOT INSTALLED")

print("\n" + "=" * 60)
