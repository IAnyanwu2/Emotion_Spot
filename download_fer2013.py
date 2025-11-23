#!/usr/bin/env python3
"""
Script to download FER2013 dataset from Kaggle
Requirements: pip install kaggle
"""

import os
import zipfile
import shutil
from pathlib import Path

def setup_fer2013_dataset():
    """Download and setup FER2013 dataset"""
    
    print("=== FER2013 Dataset Setup ===")
    
    # Create dataset directory
    dataset_dir = Path("datasets/FER2013")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Dataset directory: {dataset_dir.absolute()}")
    
    print("\n📋 Steps to download FER2013:")
    print("1. Create a Kaggle account at https://kaggle.com")
    print("2. Go to your Kaggle account settings: https://www.kaggle.com/account")
    print("3. Scroll to 'API' section and click 'Create New Token'")
    print("4. This downloads 'kaggle.json' - place it in:")
    print("   Windows: C:\\Users\\{username}\\.kaggle\\kaggle.json")
    print("   Linux/Mac: ~/.kaggle/kaggle.json")
    print("5. Run: pip install kaggle")
    print("6. Run: kaggle datasets download -d msambare/fer2013")
    print("7. Extract the zip file to the datasets/FER2013 folder")
    
    print("\n🔧 Alternative: Manual download")
    print("1. Go to: https://www.kaggle.com/datasets/msambare/fer2013")
    print("2. Click 'Download' button")
    print("3. Extract to:", dataset_dir.absolute())
    
    print("\n📁 Expected structure after extraction:")
    print("datasets/FER2013/")
    print("├── train/")
    print("│   ├── angry/")
    print("│   ├── disgust/")
    print("│   ├── fear/")
    print("│   ├── happy/")
    print("│   ├── neutral/")
    print("│   ├── sad/")
    print("│   └── surprise/")
    print("└── test/")
    print("    ├── angry/")
    print("    ├── disgust/")
    print("    ├── fear/")
    print("    ├── happy/")
    print("    ├── neutral/")
    print("    ├── sad/")
    print("    └── surprise/")

if __name__ == "__main__":
    setup_fer2013_dataset()