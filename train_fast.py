#!/usr/bin/env python3
"""
Lightweight Fast Training Script for CPU
Optimized for quick training on local machine
"""

import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from collections import Counter
import time

class LightweightEmotionCNN(nn.Module):
    """Very lightweight CNN optimized for CPU training"""
    def __init__(self, num_classes=6):
        super().__init__()
        
        # Lightweight feature extraction
        self.features = nn.Sequential(
            # Block 1 - Small filters
            nn.Conv2d(3, 16, 5, stride=2, padding=2),  # 48x48 -> 24x24
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 24x24 -> 12x12
            
            # Block 2
            nn.Conv2d(16, 32, 3, padding=1),  # 12x12 -> 12x12
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 12x12 -> 6x6
            
            # Block 3
            nn.Conv2d(32, 64, 3, padding=1),  # 6x6 -> 6x6
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # 6x6 -> 1x1
        )
        
        # Simple temporal processing
        self.temporal = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, time, height, width, channels)
        b, t, h, w, c = x.shape
        
        # Process all frames at once
        x = x.view(b * t, c, h, w)
        features = self.features(x)  # (b*t, 64, 1, 1)
        features = features.view(b * t, -1)  # (b*t, 64)
        
        # Temporal processing
        temp_features = self.temporal(features)  # (b*t, 32)
        
        # Reshape and average over time
        temp_features = temp_features.view(b, t, -1)  # (b, t, 32)
        pooled = torch.mean(temp_features, dim=1)  # (b, 32)
        
        # Classify
        output = self.classifier(pooled)
        return output

class FastRAVDESSDataset(Dataset):
    """Fast dataset loader for quick training"""
    def __init__(self, video_dirs, max_frames=15, frame_size=48):  # Reduced frames and size
        self.video_paths = []
        self.labels = []
        self.max_frames = max_frames
        self.frame_size = frame_size
        
        # Find available emotions
        available_emotions = set()
        temp_data = []
        
        for video_dir in video_dirs:
            video_path = Path(video_dir)
            if video_path.exists():
                for video_file in video_path.glob("*.mp4"):
                    parts = video_file.stem.split('-')
                    if len(parts) >= 3:
                        emotion_code = parts[2]
                        available_emotions.add(emotion_code)
                        temp_data.append((video_file, emotion_code))
        
        # Create emotion mapping
        emotion_codes = sorted(list(available_emotions))
        self.emotion_map = {code: idx for idx, code in enumerate(emotion_codes)}
        
        full_emotion_names = {
            '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
            '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
        }
        
        self.emotion_names = [full_emotion_names[code] for code in emotion_codes]
        self.num_classes = len(self.emotion_names)
        
        # Build dataset
        for video_file, emotion_code in temp_data:
            if emotion_code in self.emotion_map:
                self.video_paths.append(video_file)
                self.labels.append(self.emotion_map[emotion_code])
        
        print(f"📊 Fast dataset: {len(self.video_paths)} videos")
        print(f"🎭 Emotions: {self.emotion_names}")
        self._print_distribution()
    
    def _print_distribution(self):
        label_counts = Counter(self.labels)
        for emotion_idx, count in label_counts.items():
            print(f"   {self.emotion_names[emotion_idx]}: {count}")
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        frames = self._load_frames_fast(video_path)
        if frames is None:
            frames = np.zeros((self.max_frames, self.frame_size, self.frame_size, 3))
        
        frames_tensor = torch.from_numpy(frames).float() / 255.0
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return frames_tensor, label_tensor
    
    def _load_frames_fast(self, video_path):
        """Fast frame loading with reduced frames"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            frame_count = 0
            
            # Read every 3rd frame for speed
            while len(frames) < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % 3 == 0:  # Every 3rd frame
                    frame_small = cv2.resize(frame, (self.frame_size, self.frame_size))
                    frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                
                frame_count += 1
            
            cap.release()
            
            # Pad if needed
            while len(frames) < self.max_frames:
                frames.append(frames[-1] if frames else np.zeros((self.frame_size, self.frame_size, 3)))
            
            return np.array(frames[:self.max_frames])
            
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            return None

def train_fast():
    """Fast training for CPU"""
    print("⚡ FAST SPOT ROBOT EMOTION TRAINING")
    print("=" * 50)
    
    device = torch.device('cpu')  # Force CPU
    print("🖥️  Fast CPU training mode")
    
    # Create dataset
    video_dirs = [
        "Video_Song_Actor_02/Actor_02",
        "Video_Song_Actor_06/Actor_06"
    ]
    
    dataset = FastRAVDESSDataset(video_dirs, max_frames=15, frame_size=48)
    
    # Split
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    
    train_indices = list(range(train_size))
    test_indices = list(range(train_size, total_size))
    
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)
    
    print(f"📊 Train: {len(train_subset)}, Test: {len(test_subset)}")
    
    # Data loaders - smaller batch size
    train_loader = DataLoader(train_subset, batch_size=2, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=2, shuffle=False, num_workers=0)
    
    # Lightweight model
    model = LightweightEmotionCNN(num_classes=dataset.num_classes)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"⚡ Lightweight model: {param_count:,} parameters")
    
    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Fast training - fewer epochs
    num_epochs = 15
    best_acc = 0
    
    print("\n⚡ Starting fast training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        print(f"Epoch {epoch+1}/{num_epochs}...")
        
        for batch_idx, (videos, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Progress indicator
            if batch_idx % 20 == 0:
                print(f"   Batch {batch_idx}/{len(train_loader)}: Loss {loss.item():.3f}")
        
        train_acc = 100 * train_correct / train_total
        
        # Test
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for videos, labels in test_loader:
                outputs = model(videos)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_acc = 100 * test_correct / test_total
        
        print(f"   Train Acc: {train_acc:.1f}%, Test Acc: {test_acc:.1f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'fast_spot_model.pth')
            print(f"   🎯 New best: {best_acc:.1f}%")
        
        print("-" * 30)
    
    total_time = time.time() - start_time
    print(f"\n🎉 Training complete in {total_time/60:.1f} minutes!")
    print(f"🏆 Best accuracy: {best_acc:.1f}%")
    
    return model, dataset, best_acc

def test_fast_model():
    """Quick test of the trained model"""
    print("\n🧪 Testing fast model...")
    
    # Load dataset
    video_dirs = ["Video_Song_Actor_02/Actor_02", "Video_Song_Actor_06/Actor_06"]
    dataset = FastRAVDESSDataset(video_dirs, max_frames=15, frame_size=48)
    
    # Load model
    model = LightweightEmotionCNN(num_classes=dataset.num_classes)
    model.load_state_dict(torch.load('fast_spot_model.pth', map_location='cpu'))
    model.eval()
    
    # Test one video
    test_video = "Video_Song_Actor_02/Actor_02/01-02-03-01-01-01-02.mp4"  # Happy
    if Path(test_video).exists():
        frames = dataset._load_frames_fast(Path(test_video))
        if frames is not None:
            video_tensor = torch.from_numpy(frames).float() / 255.0
            video_tensor = video_tensor.unsqueeze(0)
            
            with torch.no_grad():
                output = model(video_tensor)
                probs = torch.softmax(output, dim=1)
                pred_idx = torch.argmax(probs)
                confidence = probs[0, pred_idx].item()
            
            predicted = dataset.emotion_names[pred_idx]
            print(f"   📹 Video: Happy test")
            print(f"   🤖 Predicted: {predicted} ({confidence:.3f})")

if __name__ == "__main__":
    model, dataset, accuracy = train_fast()
    test_fast_model()
    
    print(f"\n⚡ FAST TRAINING RESULTS:")
    print(f"🏆 Final accuracy: {accuracy:.1f}%")
    print(f"💾 Model saved: fast_spot_model.pth")
    print(f"🤖 Ready for Spot robot!")