#!/usr/bin/env python3
"""
Simple Local Test for RAVDESS Video Processing
Step 1: Test basic video loading and processing on your machine
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt

def decode_ravdess_filename(filename):
    """Decode RAVDESS filename to extract emotion and other info"""
    parts = filename.stem.split('-')
    if len(parts) != 7:
        return None
    
    emotion_map = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }
    
    return {
        'modality': parts[0],  # 01=video-only, 02=audio-video
        'vocal_channel': parts[1],  # 01=speech, 02=song
        'emotion': emotion_map.get(parts[2], 'unknown'),
        'emotion_code': parts[2],
        'intensity': parts[3],  # 01=normal, 02=strong
        'statement': parts[4],
        'repetition': parts[5],
        'actor': parts[6].split('.')[0]
    }

def test_video_loading():
    """Test 1: Can we load and process videos?"""
    print("🎬 Testing Video Loading...")
    
    # Find first video file
    video_dir = Path("c:/Users/Ikean/RJCMA/Video_Song_Actor_02/Actor_02")
    video_files = list(video_dir.glob("*.mp4"))
    
    if not video_files:
        print("❌ No video files found!")
        return False
    
    # Test first video
    video_path = video_files[0]
    print(f"📹 Testing: {video_path.name}")
    
    # Decode filename
    info = decode_ravdess_filename(video_path)
    if info:
        print(f"   Emotion: {info['emotion']}")
        print(f"   Actor: {info['actor']}")
        print(f"   Intensity: {'Strong' if info['intensity'] == '02' else 'Normal'}")
    
    # Try to load video
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print("❌ Cannot open video file")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"   📊 Video Properties:")
        print(f"      - FPS: {fps}")
        print(f"      - Frames: {frame_count}")
        print(f"      - Size: {width}x{height}")
        print(f"      - Duration: {frame_count/fps:.2f} seconds")
        
        # Read first frame
        ret, frame = cap.read()
        if ret:
            print(f"   ✅ Successfully read frame: {frame.shape}")
            
            # Convert to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize for model (if needed)
            frame_resized = cv2.resize(frame_rgb, (224, 224))
            print(f"   ✅ Resized frame: {frame_resized.shape}")
            
            cap.release()
            return True, video_path, info
        else:
            print("❌ Cannot read frame")
            cap.release()
            return False
            
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        return False

def test_audio_extraction():
    """Test 2: Can we extract audio from video?"""
    print("\n🎵 Testing Audio Extraction...")
    
    try:
        import librosa
        print("   ✅ librosa available")
        
        # Test audio extraction from video
        video_dir = Path("c:/Users/Ikean/RJCMA/Video_Song_Actor_02/Actor_02")
        video_files = list(video_dir.glob("*.mp4"))
        
        if video_files:
            video_path = video_files[0]
            print(f"   🎬 Extracting audio from: {video_path.name}")
            
            # Load audio using librosa
            audio, sr = librosa.load(str(video_path), sr=16000)  # 16kHz for Spot robot
            print(f"   ✅ Audio loaded: {audio.shape}, Sample rate: {sr}")
            print(f"   📊 Audio duration: {len(audio)/sr:.2f} seconds")
            
            # Extract MFCC features (for emotion recognition)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            print(f"   ✅ MFCC features: {mfcc.shape}")
            
            return True, audio, mfcc
        else:
            print("   ❌ No video files to test")
            return False
            
    except ImportError:
        print("   ❌ librosa not available")
        return False
    except Exception as e:
        print(f"   ❌ Error extracting audio: {e}")
        return False

def test_torch_processing():
    """Test 3: Can we create torch tensors for model input?"""
    print("\n🔥 Testing PyTorch Processing...")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   🖥️  Device: {device}")
        
        # Create sample video tensor (batch_size=1, time=30, height=224, width=224, channels=3)
        video_tensor = torch.randn(1, 30, 224, 224, 3)
        print(f"   ✅ Video tensor: {video_tensor.shape}")
        
        # Create sample audio tensor (batch_size=1, time=500, features=13)
        audio_tensor = torch.randn(1, 500, 13)  # MFCC features
        print(f"   ✅ Audio tensor: {audio_tensor.shape}")
        
        # Move to device
        video_tensor = video_tensor.to(device)
        audio_tensor = audio_tensor.to(device)
        print(f"   ✅ Tensors moved to {device}")
        
        # Simple processing test
        video_processed = torch.mean(video_tensor, dim=1)  # Average over time
        audio_processed = torch.mean(audio_tensor, dim=1)  # Average over time
        
        print(f"   ✅ Processed shapes: video={video_processed.shape}, audio={audio_processed.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error in torch processing: {e}")
        return False

def test_all_videos():
    """Test 4: Check all available videos"""
    print("\n📁 Checking All Available Videos...")
    
    video_dirs = [
        "c:/Users/Ikean/RJCMA/Video_Song_Actor_02/Actor_02",
        "c:/Users/Ikean/RJCMA/Video_Song_Actor_06/Actor_06"
    ]
    
    total_videos = 0
    emotion_counts = {}
    
    for video_dir in video_dirs:
        video_path = Path(video_dir)
        if video_path.exists():
            videos = list(video_path.glob("*.mp4"))
            print(f"   📂 {video_path.name}: {len(videos)} videos")
            total_videos += len(videos)
            
            # Count emotions
            for video in videos[:5]:  # Check first 5
                info = decode_ravdess_filename(video)
                if info:
                    emotion = info['emotion']
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    print(f"   📊 Total videos: {total_videos}")
    print(f"   🎭 Emotions found: {emotion_counts}")
    
    return total_videos > 0

def main():
    """Run all tests"""
    print("🧪 LOCAL MACHINE TESTING - RAVDESS VIDEO DATA")
    print("=" * 60)
    
    # Test 1: Video loading
    video_test = test_video_loading()
    if not video_test:
        print("\n❌ Video loading failed. Check your video files.")
        return
    
    # Test 2: Audio extraction
    audio_test = test_audio_extraction()
    
    # Test 3: PyTorch processing
    torch_test = test_torch_processing()
    
    # Test 4: Check all videos
    all_videos_test = test_all_videos()
    
    print("\n" + "=" * 60)
    print("📋 TEST RESULTS:")
    print(f"   Video Loading: {'✅' if video_test else '❌'}")
    print(f"   Audio Extraction: {'✅' if audio_test else '❌'}")
    print(f"   PyTorch Processing: {'✅' if torch_test else '❌'}")
    print(f"   All Videos Check: {'✅' if all_videos_test else '❌'}")
    
    if all([video_test, audio_test, torch_test, all_videos_test]):
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Ready for model testing!")
        print("\nNext step: Run 'python test_model_simple.py'")
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    main()