#!/usr/bin/env python3
"""
Fixed Training Script with Better Accuracy and Error Handling
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

class ImprovedEmotionCNN(nn.Module):
    """Improved CNN with better architecture for emotion recognition"""
    def __init__(self, num_classes=6, input_size=48):  # 6 classes found in your data
        super().__init__()
        self.num_classes = num_classes
        
        # Improved feature extraction
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Block 2  
            nn.Conv2d(64, 128, 5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        
        # Temporal processing with attention
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(512*4, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        b, t, h, w, c = x.shape
        
        # Extract spatial features
        x = x.view(b * t, c, h, w)
        features = self.features(x)  # (b*t, 512, 2, 2)
        features = features.view(b * t, -1)  # (b*t, 512*4)
        
        # Reshape for temporal processing
        features = features.view(b, t, -1).transpose(1, 2)  # (b, 512*4, t)
        
        # Temporal convolution
        temp_features = self.temporal_conv(features)  # (b, 128, t)
        temp_features = temp_features.transpose(1, 2)  # (b, t, 128)
        
        # Attention mechanism
        attention_weights = self.attention(temp_features)  # (b, t, 1)
        attended_features = temp_features * attention_weights  # (b, t, 128)
        
        # Global pooling with attention
        pooled_features = torch.sum(attended_features, dim=1)  # (b, 128)
        
        # Classification
        output = self.classifier(pooled_features)
        return output

class FixedRAVDESSDataset(Dataset):
    """Fixed dataset with proper emotion mapping for available classes"""
    def __init__(self, video_dirs, max_frames=30, frame_size=64, augment=True):
        self.video_paths = []
        self.labels = []
        self.max_frames = max_frames
        self.frame_size = frame_size
        self.augment = augment
        
        # First pass: find all available emotions
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
        
        # Create mapping for available emotions only
        emotion_codes = sorted(list(available_emotions))
        self.emotion_map = {code: idx for idx, code in enumerate(emotion_codes)}
        
        # Full emotion names for reference
        full_emotion_names = {
            '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
            '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
        }
        
        self.emotion_names = [full_emotion_names[code] for code in emotion_codes]
        self.num_classes = len(self.emotion_names)
        
        # Build final dataset
        for video_file, emotion_code in temp_data:
            if emotion_code in self.emotion_map:
                self.video_paths.append(video_file)
                self.labels.append(self.emotion_map[emotion_code])
        
        print(f"📊 Dataset loaded: {len(self.video_paths)} videos")
        print(f"🎭 Available emotions ({self.num_classes}): {self.emotion_names}")
        self._print_class_distribution()
    
    def _print_class_distribution(self):
        """Print distribution of emotions in dataset"""
        label_counts = Counter(self.labels)
        print("📈 Emotion distribution:")
        for emotion_idx, count in label_counts.items():
            emotion_name = self.emotion_names[emotion_idx]
            print(f"   {emotion_name}: {count} videos")
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        frames = self._load_video_frames(video_path)
        if frames is None:
            frames = np.zeros((self.max_frames, self.frame_size, self.frame_size, 3))
        
        # Data augmentation during training
        if self.augment:
            frames = self._augment_frames(frames)
        
        frames_tensor = torch.from_numpy(frames).float() / 255.0
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return frames_tensor, label_tensor
    
    def _load_video_frames(self, video_path):
        """Load video frames with better sampling"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                cap.release()
                return None
            
            # Sample frames uniformly across the video
            frame_indices = np.linspace(0, total_frames-1, self.max_frames, dtype=int)
            frames = []
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame_resized = cv2.resize(frame, (self.frame_size, self.frame_size))
                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                else:
                    # Duplicate last frame if read fails
                    if frames:
                        frames.append(frames[-1])
                    else:
                        frames.append(np.zeros((self.frame_size, self.frame_size, 3)))
            
            cap.release()
            return np.array(frames)
            
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            return None
    
    def _augment_frames(self, frames):
        """Simple data augmentation"""
        if np.random.random() > 0.5:
            # Random horizontal flip - fix negative stride issue
            frames = frames[:, :, ::-1, :].copy()
        
        if np.random.random() > 0.7:
            # Random brightness adjustment
            brightness_factor = np.random.uniform(0.8, 1.2)
            frames = np.clip(frames * brightness_factor, 0, 255)
        
        return frames

def train_improved_model():
    """Train the improved model"""
    print("🚀 IMPROVED SPOT ROBOT EMOTION RECOGNITION")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Training on: {device}")
    
    # Create dataset
    video_dirs = [
        "Video_Song_Actor_02/Actor_02",
        "Video_Song_Actor_06/Actor_06"
    ]
    
    # Training dataset with augmentation
    train_dataset = FixedRAVDESSDataset(video_dirs, max_frames=30, frame_size=64, augment=True)
    num_classes = train_dataset.num_classes
    
    # Test dataset without augmentation
    test_dataset = FixedRAVDESSDataset(video_dirs, max_frames=30, frame_size=64, augment=False)
    
    # Split dataset
    total_size = len(train_dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    
    train_indices = list(range(train_size))
    test_indices = list(range(train_size, total_size))
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)
    
    print(f"📊 Train samples: {len(train_subset)}")
    print(f"📊 Test samples: {len(test_subset)}")
    
    # Data loaders
    train_loader = DataLoader(train_subset, batch_size=4, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=4, shuffle=False, num_workers=0)
    
    # Improved model
    model = ImprovedEmotionCNN(num_classes=num_classes, input_size=64).to(device)
    print(f"🤖 Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Better optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    # Training loop
    num_epochs = 30
    best_accuracy = 0
    
    print("\n🏋️  Starting improved training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (videos, labels) in enumerate(train_loader):
            videos, labels = videos.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        
        # Test accuracy
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for videos, labels in test_loader:
                videos, labels = videos.to(device), labels.to(device)
                outputs = model(videos)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_accuracy = 100 * test_correct / test_total
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"   Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.2f}%")
        print(f"   Test Acc: {test_accuracy:.2f}%")
        
        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'best_spot_emotion_model.pth')
            print(f"   🎯 New best model saved! Accuracy: {best_accuracy:.2f}%")
        
        print("-" * 50)
    
    print(f"\n🎉 Training complete!")
    print(f"🏆 Best test accuracy: {best_accuracy:.2f}%")
    
    return model, train_dataset.emotion_names, best_accuracy

def test_final_model():
    """Test the final trained model"""
    print("\n🧪 Testing Final Model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset to get emotion names
    video_dirs = ["Video_Song_Actor_02/Actor_02", "Video_Song_Actor_06/Actor_06"]
    dataset = FixedRAVDESSDataset(video_dirs, max_frames=30, frame_size=64, augment=False)
    
    # Load best model
    model = ImprovedEmotionCNN(num_classes=dataset.num_classes, input_size=64).to(device)
    model.load_state_dict(torch.load('best_spot_emotion_model.pth', map_location=device))
    model.eval()
    
    # Test on a few videos
    test_videos = [
        "Video_Song_Actor_02/Actor_02/01-02-03-01-01-01-02.mp4",  # Happy
        "Video_Song_Actor_02/Actor_02/01-02-05-01-01-01-02.mp4",  # Angry
        "Video_Song_Actor_02/Actor_02/01-02-01-01-01-01-02.mp4",  # Neutral
    ]
    
    for video_path in test_videos:
        if Path(video_path).exists():
            # Get true emotion
            parts = Path(video_path).stem.split('-')
            emotion_map = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
                          '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
            true_emotion = emotion_map.get(parts[2], 'unknown')
            
            # Load and predict
            temp_dataset = FixedRAVDESSDataset([Path(video_path).parent], max_frames=30, frame_size=64, augment=False)
            for i, (video_tensor, _) in enumerate(temp_dataset):
                if temp_dataset.video_paths[i].name == Path(video_path).name:
                    video_tensor = video_tensor.unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        output = model(video_tensor)
                        probabilities = torch.softmax(output, dim=1)
                        predicted_idx = torch.argmax(probabilities, dim=1)
                        confidence = probabilities[0, predicted_idx].item()
                    
                    predicted_emotion = dataset.emotion_names[predicted_idx.item()]
                    
                    print(f"   📹 {Path(video_path).name}")
                    print(f"   🎯 True: {true_emotion}")
                    print(f"   🤖 Predicted: {predicted_emotion} ({confidence:.3f})")
                    print(f"   {'✅' if predicted_emotion == true_emotion else '❌'} Match")
                    print()
                    break

if __name__ == "__main__":
    # Train improved model
    model, emotion_names, best_acc = train_improved_model()
    
    # Test the final model
    test_final_model()
    
    print(f"\n🎉 FINAL RESULTS:")
    print(f"🏆 Best Accuracy: {best_acc:.2f}%")
    print(f"🎭 Emotions: {emotion_names}")
    print(f"💾 Model saved as: best_spot_emotion_model.pth")
    print(f"🤖 Ready for Spot robot deployment!")