#!/usr/bin/env python3
"""
Multimodal Dataset Setup for Spot Robot Emotion Recognition
Supports: RAVDESS, CREMA-D, and maintains FER2013 compatibility
"""

import os
import urllib.request
import zipfile
from pathlib import Path

def setup_ravdess_dataset():
    """Setup RAVDESS dataset for video + audio emotion recognition"""
    print("🎭 Setting up RAVDESS Dataset for Spot Robot...")
    print("=" * 60)
    
    dataset_dir = Path("datasets/RAVDESS")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print("📋 RAVDESS Dataset Information:")
    print("- 24 professional actors (12 female, 12 male)")
    print("- 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised")
    print("- Video + Audio + Speech")
    print("- Perfect for robot interaction scenarios")
    
    print("\n📥 Download Instructions:")
    print("1. Go to: https://zenodo.org/record/1188976")
    print("2. Download these files:")
    print("   - Video_Speech_Actor_*.zip (for video+audio)")
    print("   - Audio_Speech_Actors_*.zip (for audio-only)")
    print("3. Extract to:", dataset_dir.absolute())
    
    print("\n🤖 Why RAVDESS is perfect for Spot:")
    print("- Natural speech patterns (like humans talking to robot)")
    print("- Clear facial expressions (for camera input)")
    print("- Audio emotional cues (for microphone input)")
    print("- Controlled lighting (similar to indoor robot use)")
    
    return dataset_dir

def setup_crema_d_dataset():
    """Setup CREMA-D dataset"""
    print("\n🎬 Setting up CREMA-D Dataset...")
    print("=" * 60)
    
    dataset_dir = Path("datasets/CREMA-D")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print("📋 CREMA-D Dataset Information:")
    print("- 91 actors (48 male, 43 female)")
    print("- 6 emotions: angry, disgust, fear, happy, neutral, sad")
    print("- More diverse ethnic backgrounds")
    print("- Video + Audio")
    
    print("\n📥 Download Instructions:")
    print("1. Go to: https://github.com/CheyneyComputerScience/CREMA-D")
    print("2. Follow their download instructions")
    print("3. Extract to:", dataset_dir.absolute())
    
    return dataset_dir

def create_spot_robot_config():
    """Create configuration optimized for Spot robot deployment"""
    config = {
        # Spot Robot specific settings
        "robot": {
            "name": "Spot",
            "camera_resolution": (640, 480),  # Spot's camera resolution
            "audio_sample_rate": 16000,       # Spot's microphone
            "inference_fps": 10,              # Real-time inference rate
            "emotion_confidence_threshold": 0.7,
        },
        
        # Multimodal settings
        "modalities": {
            "video": True,     # Use Spot's cameras
            "audio": True,     # Use Spot's microphones  
            "text": False,     # Optional: speech-to-text
        },
        
        # Model optimization for edge deployment
        "model": {
            "backbone": "mobilenet_v3",  # Lightweight for robot
            "input_size": (224, 224),    # Efficient input size
            "batch_size": 1,             # Real-time single inference
            "precision": "fp16",         # Half precision for speed
        },
        
        # Real-time processing
        "processing": {
            "buffer_size": 30,           # 3 seconds at 10fps
            "overlap": 0.5,              # Smooth transitions
            "emotion_smoothing": True,   # Avoid emotion flickering
        }
    }
    
    return config

def setup_for_spot_robot():
    """Complete setup for Spot robot emotion recognition"""
    print("🤖 SPOT ROBOT EMOTION RECOGNITION SETUP")
    print("=" * 70)
    
    # Create directories
    base_dir = Path("datasets")
    base_dir.mkdir(exist_ok=True)
    
    models_dir = Path("models_spot")
    models_dir.mkdir(exist_ok=True)
    
    # Setup datasets
    ravdess_dir = setup_ravdess_dataset()
    crema_d_dir = setup_crema_d_dataset()
    
    # Create Spot-specific config
    config = create_spot_robot_config()
    
    print("\n🔧 Spot Robot Integration Plan:")
    print("=" * 50)
    print("1. Data Collection:")
    print("   - Use Spot's front cameras for facial expression")
    print("   - Use Spot's microphones for voice emotion")
    print("   - Real-time processing at 10 FPS")
    
    print("\n2. Model Architecture:")
    print("   - Video branch: MobileNet backbone (lightweight)")
    print("   - Audio branch: 1D CNN for MFCC features")
    print("   - Fusion: Cross-modal attention (your RCMA model)")
    
    print("\n3. Robot Deployment:")
    print("   - Edge inference on Spot's compute")
    print("   - Low latency emotion detection")
    print("   - Emotion-based behavior triggers")
    
    print("\n4. Use Cases:")
    print("   - Security: Detect agitated/threatening behavior")
    print("   - Healthcare: Monitor patient emotional state") 
    print("   - Entertainment: Respond to user emotions")
    print("   - Research: Collect human-robot interaction data")
    
    print(f"\n📁 Setup complete! Datasets will be in: {base_dir.absolute()}")
    
    return {
        "ravdess": ravdess_dir,
        "crema_d": crema_d_dir,
        "config": config
    }

if __name__ == "__main__":
    setup_info = setup_for_spot_robot()
    
    print("\n✅ Next Steps:")
    print("1. Download the recommended datasets")
    print("2. Run: python test_multimodal.py")
    print("3. Train multimodal model")
    print("4. Deploy to Spot robot")