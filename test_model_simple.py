#!/usr/bin/env python3
"""
Step 2: Test Model Components
Check if the RCMA model can be loaded and run basic inference
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.getcwd())

def test_basic_imports():
    """Test if we can import the project modules"""
    print("📦 Testing Project Imports...")
    
    try:
        # Test basic config imports
        from configs import config
        print("   ✅ configs.py imported")
        
        # Test if we can access config values
        print(f"   📊 Video embedding dim: {config.get('feature_dimension', {}).get('video', 'Not found')}")
        
    except ImportError as e:
        print(f"   ❌ Config import failed: {e}")
        return False
    except Exception as e:
        print(f"   ⚠️  Config warning: {e}")
    
    try:
        # Test base imports
        from base.utils import load_pickle
        print("   ✅ base.utils imported")
        
    except ImportError as e:
        print(f"   ❌ Base utils import failed: {e}")
        return False
    
    try:
        # Test model imports
        from models.model import RCMA
        print("   ✅ RCMA model imported")
        return True
        
    except ImportError as e:
        print(f"   ❌ RCMA model import failed: {e}")
        print(f"   💡 This might be due to missing dependencies")
        return False

def test_model_creation():
    """Test if we can create the RCMA model"""
    print("\n🤖 Testing Model Creation...")
    
    try:
        from models.model import RCMA
        
        # Create a simple RCMA model for testing
        # These are simplified parameters for testing
        model_args = {
            'num_heads': 2,
            'modal_dim': 32,
            'modality': ['video'],  # Start with video only
            'feature_dimension': {'video': (48, 48, 3)},
            'continuous_label_dim': 7,  # 7 emotions for RAVDESS
            'tcn_kernel_size': 3
        }
        
        print("   🏗️  Creating RCMA model...")
        print(f"   📋 Model args: {model_args}")
        
        # This might fail - we'll catch and diagnose
        model = RCMA(model_args)
        print("   ✅ RCMA model created successfully!")
        
        # Test model forward pass
        batch_size = 1
        seq_len = 30
        height, width, channels = 48, 48, 3
        
        # Create dummy input
        dummy_input = {
            'video': torch.randn(batch_size, seq_len, height, width, channels)
        }
        
        print(f"   🧪 Testing forward pass with input shape: {dummy_input['video'].shape}")
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
            print(f"   ✅ Forward pass successful! Output shape: {output.shape if hasattr(output, 'shape') else type(output)}")
        
        return True, model
        
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        print(f"   🔍 Error type: {type(e).__name__}")
        return False, None

def test_simple_video_processing():
    """Test processing a real video file"""
    print("\n🎬 Testing Real Video Processing...")
    
    try:
        import cv2
        
        # Load a real video
        video_path = Path("Video_Song_Actor_02/Actor_02/01-02-01-01-01-01-02.mp4")
        
        if not video_path.exists():
            print(f"   ❌ Video file not found: {video_path}")
            return False
        
        print(f"   📹 Processing: {video_path.name}")
        
        # Load video
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        # Read first 30 frames (1 second at 30fps)
        for i in range(30):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize to model input size
            frame_resized = cv2.resize(frame, (48, 48))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        cap.release()
        
        if len(frames) < 10:
            print(f"   ⚠️  Only got {len(frames)} frames")
        
        # Convert to tensor
        frames_array = np.array(frames)
        video_tensor = torch.from_numpy(frames_array).float()
        video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension
        
        print(f"   ✅ Video tensor created: {video_tensor.shape}")
        print(f"   📊 Tensor range: [{video_tensor.min():.2f}, {video_tensor.max():.2f}]")
        
        # Normalize to [0, 1] range
        video_tensor = video_tensor / 255.0
        print(f"   ✅ Normalized tensor range: [{video_tensor.min():.2f}, {video_tensor.max():.2f}]")
        
        return True, video_tensor
        
    except Exception as e:
        print(f"   ❌ Video processing failed: {e}")
        return False, None

def test_ravdess_emotions():
    """Test emotion label mapping for RAVDESS"""
    print("\n🎭 Testing RAVDESS Emotion Mapping...")
    
    # RAVDESS emotion mapping
    emotion_map = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }
    
    print("   📋 RAVDESS Emotions:")
    for code, emotion in emotion_map.items():
        print(f"      {code}: {emotion}")
    
    # Test filename parsing
    test_files = [
        "01-02-03-01-01-01-02.mp4",  # Happy
        "01-02-05-02-01-01-02.mp4",  # Angry, strong intensity
        "01-02-06-01-01-01-02.mp4",  # Fearful
    ]
    
    print("\n   🧪 Testing filename parsing:")
    for filename in test_files:
        parts = filename.replace('.mp4', '').split('-')
        if len(parts) == 7:
            emotion_code = parts[2]
            emotion = emotion_map.get(emotion_code, 'unknown')
            intensity = 'Strong' if parts[3] == '02' else 'Normal'
            print(f"      {filename} → {emotion} ({intensity})")
    
    return True

def main():
    """Run all model tests"""
    print("🤖 MODEL COMPONENT TESTING")
    print("=" * 60)
    
    # Test 1: Basic imports
    imports_ok = test_basic_imports()
    
    # Test 2: Model creation
    if imports_ok:
        model_ok, model = test_model_creation()
    else:
        print("   ⏭️  Skipping model creation due to import failures")
        model_ok = False
        model = None
    
    # Test 3: Video processing
    video_ok, video_tensor = test_simple_video_processing()
    
    # Test 4: Emotion mapping
    emotion_ok = test_ravdess_emotions()
    
    print("\n" + "=" * 60)
    print("📋 MODEL TEST RESULTS:")
    print(f"   Project Imports: {'✅' if imports_ok else '❌'}")
    print(f"   Model Creation: {'✅' if model_ok else '❌'}")
    print(f"   Video Processing: {'✅' if video_ok else '❌'}")
    print(f"   Emotion Mapping: {'✅' if emotion_ok else '❌'}")
    
    # If everything works, test full pipeline
    if all([imports_ok, model_ok, video_ok]) and model is not None and video_tensor is not None:
        print("\n🔗 Testing Full Pipeline...")
        try:
            model.eval()
            with torch.no_grad():
                # This might not work yet, but let's try
                dummy_input = {'video': video_tensor}
                output = model(dummy_input)
                print(f"   ✅ Full pipeline test successful!")
                print(f"   📊 Model output: {output.shape if hasattr(output, 'shape') else type(output)}")
        except Exception as e:
            print(f"   ⚠️  Full pipeline failed: {e}")
            print("   💡 This is expected - model needs proper configuration")
    
    if imports_ok and video_ok:
        print("\n🎉 Basic components working!")
        print("✅ Ready for next step: Model configuration")
    else:
        print("\n⚠️  Some components need fixing")
        print("💡 Let's address the import issues first")

if __name__ == "__main__":
    main()