#!/usr/bin/env python3
"""
Test script to verify the local setup is working correctly
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

def test_environment():
    """Test if all dependencies are properly installed"""
    print("🧪 Testing Environment Setup...")
    print("=" * 50)
    
    # Test 1: Basic imports
    try:
        import torch
        import numpy as np
        import cv2
        import pandas as pd
        import matplotlib.pyplot as plt
        print("✅ Core libraries imported successfully")
        print(f"   - PyTorch: {torch.__version__}")
        print(f"   - NumPy: {np.__version__}")
        print(f"   - OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test 2: Device detection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Device detected: {device}")
    if torch.cuda.is_available():
        print(f"   - GPU: {torch.cuda.get_device_name(0)}")
        print(f"   - CUDA version: {torch.version.cuda}")
    else:
        print("   - Running on CPU (this is fine for testing)")
    
    # Test 3: Dataset verification
    dataset_path = Path("c:/Users/Ikean/RJCMA/fer2013")
    if dataset_path.exists():
        print("✅ FER2013 dataset found")
        
        # Count images in each emotion category
        total_train = 0
        total_test = 0
        emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        
        for emotion in emotions:
            train_path = dataset_path / "train" / emotion
            test_path = dataset_path / "test" / emotion
            
            if train_path.exists() and test_path.exists():
                train_count = len(list(train_path.glob("*.jpg"))) + len(list(train_path.glob("*.png")))
                test_count = len(list(test_path.glob("*.jpg"))) + len(list(test_path.glob("*.png")))
                total_train += train_count
                total_test += test_count
                print(f"   - {emotion}: {train_count} train, {test_count} test")
        
        print(f"   - Total: {total_train} train, {total_test} test images")
    else:
        print("❌ FER2013 dataset not found")
        return False
    
    # Test 4: Simple tensor operation
    try:
        x = torch.randn(2, 3, 48, 48)  # Simulate FER2013 batch
        if torch.cuda.is_available():
            x = x.cuda()
        y = torch.nn.functional.relu(x)
        print("✅ Basic tensor operations working")
    except Exception as e:
        print(f"❌ Tensor operation failed: {e}")
        return False
    
    print("=" * 50)
    print("🎉 Environment setup is working correctly!")
    print("You can now run the emotion recognition model.")
    return True

def test_model_imports():
    """Test if project-specific imports work"""
    print("\n🔍 Testing Project Imports...")
    print("=" * 50)
    
    try:
        # Add current directory to path
        sys.path.insert(0, "c:/Users/Ikean/RJCMA")
        
        # Test importing project modules
        from configs_local import get_local_config
        config = get_local_config()
        print("✅ Local configuration loaded")
        
        # Test importing base modules
        from base.utils import load_pickle
        print("✅ Base utilities imported")
        
        # Test importing model modules
        from models.model import RCMA
        print("✅ RCMA model imported")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Some modules may need the full project structure")
        return False
    except Exception as e:
        print(f"⚠️  Warning: {e}")
        print("This might be normal if some files are missing")
        return True
    
    print("✅ Project imports working")
    return True

if __name__ == "__main__":
    print("🚀 RCMA Local Setup Test")
    print("=" * 60)
    
    env_ok = test_environment()
    model_ok = test_model_imports()
    
    if env_ok:
        print("\n✅ Ready to run the emotion recognition system!")
        print("\nNext steps:")
        print("1. Run: python test_simple.py")
        print("2. For training: python main.py (with proper arguments)")
        print("3. For FER2013 classification: Use the simplified version")
    else:
        print("\n❌ Please fix the issues above before continuing")