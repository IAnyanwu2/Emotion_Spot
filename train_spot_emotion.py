#!/usr/bin/env python3
"""
RAVDESS Emotion Recognition Training for Spot Robot
Train a simple but effective emotion classifier on your video data
"""

import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import time

class SimpleEmotionCNN(nn.Module):
    """Lightweight CNN for Spot robot deployment"""
    def __init__(self, num_classes=8, input_size=48):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Lightweight CNN backbone
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 48x48 -> 24x24
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 24x24 -> 12x12
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 12x12 -> 6x6
            
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )
        
        # Temporal processing
        self.temporal_lstm = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),  # 128*2 from bidirectional LSTM
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, time, height, width, channels)
        b, t, h, w, c = x.shape
        
        # Reshape to process all frames
        x = x.view(b * t, c, h, w)
        
        # Extract spatial features
        features = self.features(x)  # (batch*time, 256, 1, 1)
        features = features.view(b * t, -1)  # (batch*time, 256)
        
        # Reshape for temporal processing
        features = features.view(b, t, -1)  # (batch, time, 256)
        
        # LSTM for temporal modeling
        lstm_out, _ = self.temporal_lstm(features)  # (batch, time, 256)
        
        # Average pooling over time
        temporal_features = torch.mean(lstm_out, dim=1)  # (batch, 256)
        
        # Final classification
        output = self.classifier(temporal_features)
        return output

class RAVDESSDataset(Dataset):
    """Dataset class for RAVDESS video data"""
    def __init__(self, video_dirs, max_frames=30, frame_size=48, transform=None):
        self.video_paths = []
        self.labels = []
        self.max_frames = max_frames
        self.frame_size = frame_size
        self.transform = transform
        
        # Emotion mapping
        self.emotion_map = {
            '01': 0, '02': 1, '03': 2, '04': 3,
            '05': 4, '06': 5, '07': 6, '08': 7
        }
        
        self.emotion_names = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        
        # Collect all video files
        for video_dir in video_dirs:
            video_path = Path(video_dir)
            if video_path.exists():
                for video_file in video_path.glob("*.mp4"):
                    # Parse emotion from filename
                    parts = video_file.stem.split('-')
                    if len(parts) >= 3:
                        emotion_code = parts[2]
                        if emotion_code in self.emotion_map:
                            self.video_paths.append(video_file)
                            self.labels.append(self.emotion_map[emotion_code])
        
        print(f"📊 Dataset loaded: {len(self.video_paths)} videos")
        self._print_class_distribution()
    
    def _print_class_distribution(self):
        """Print distribution of emotions in dataset"""
        from collections import Counter
        label_counts = Counter(self.labels)
        print("🎭 Emotion distribution:")
        for emotion_idx, count in label_counts.items():
            emotion_name = self.emotion_names[emotion_idx]
            print(f"   {emotion_name}: {count} videos")
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load video frames
        frames = self._load_video_frames(video_path)
        if frames is None:
            # Return dummy data if video loading fails
            frames = np.zeros((self.max_frames, self.frame_size, self.frame_size, 3))
        
        # Convert to tensor
        frames_tensor = torch.from_numpy(frames).float() / 255.0
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return frames_tensor, label_tensor
    
    def _load_video_frames(self, video_path):
        """Load and preprocess video frames"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            
            # Read frames
            frame_count = 0
            while len(frames) < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize and convert to RGB
                frame_resized = cv2.resize(frame, (self.frame_size, self.frame_size))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                frame_count += 1
            
            cap.release()
            
            # Pad with last frame if needed
            while len(frames) < self.max_frames:
                frames.append(frames[-1] if frames else np.zeros((self.frame_size, self.frame_size, 3)))
            
            return np.array(frames[:self.max_frames])
            
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            return None

def train_model():
    """Train the emotion recognition model"""
    print("🚀 TRAINING SPOT ROBOT EMOTION RECOGNITION MODEL")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Training on: {device}")
    
    # Dataset
    video_dirs = [
        "Video_Song_Actor_02/Actor_02",
        "Video_Song_Actor_06/Actor_06"
    ]
    
    dataset = RAVDESSDataset(video_dirs, max_frames=30, frame_size=48)
    
    # Split dataset (80% train, 20% test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    print(f"📊 Train samples: {len(train_dataset)}")
    print(f"📊 Test samples: {len(test_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    # Model
    model = SimpleEmotionCNN(num_classes=8, input_size=48).to(device)
    print(f"🤖 Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training loop
    num_epochs = 20
    train_losses = []
    train_accuracies = []
    
    print("\n🏋️  Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        start_time = time.time()
        
        for batch_idx, (videos, labels) in enumerate(train_loader):
            videos, labels = videos.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"   Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct_predictions / total_predictions
        epoch_time = time.time() - start_time
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"   Loss: {epoch_loss:.4f}")
        print(f"   Accuracy: {epoch_accuracy:.2f}%")
        print(f"   Time: {epoch_time:.2f}s")
        print("-" * 40)
    
    # Test the model
    print("\n🧪 Testing model...")
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for videos, labels in test_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_accuracy = accuracy_score(all_labels, all_predictions)
    print(f"🎯 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Classification report
    emotion_names = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    print("\n📊 Detailed Results:")
    print(classification_report(all_labels, all_predictions, target_names=emotion_names))
    
    # Save model
    torch.save(model.state_dict(), 'spot_emotion_model.pth')
    print("\n💾 Model saved as 'spot_emotion_model.pth'")
    
    return model, train_losses, train_accuracies, test_accuracy

def test_single_video(model_path='spot_emotion_model.pth'):
    """Test the trained model on a single video"""
    print("\n🎬 Testing trained model on single video...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = SimpleEmotionCNN(num_classes=8, input_size=48).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Test video
    video_path = Path("Video_Song_Actor_02/Actor_02/01-02-03-01-01-01-02.mp4")  # Happy
    
    if not video_path.exists():
        print("❌ Test video not found")
        return
    
    # Get true emotion
    parts = video_path.stem.split('-')
    emotion_map = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
                   '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
    true_emotion = emotion_map.get(parts[2], 'unknown')
    
    # Load and process video
    dataset = RAVDESSDataset([video_path.parent], max_frames=30, frame_size=48)
    video_tensor, _ = dataset[0]  # Get first video
    video_tensor = video_tensor.unsqueeze(0).to(device)  # Add batch dimension
    
    # Predict
    with torch.no_grad():
        output = model(video_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1)
        confidence = probabilities[0, predicted_idx].item()
    
    emotion_names = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    predicted_emotion = emotion_names[predicted_idx.item()]
    
    print(f"   📹 Video: {video_path.name}")
    print(f"   🎯 True emotion: {true_emotion}")
    print(f"   🤖 Predicted: {predicted_emotion}")
    print(f"   📊 Confidence: {confidence:.3f}")
    print(f"   {'✅' if predicted_emotion == true_emotion else '❌'} Match: {predicted_emotion == true_emotion}")

if __name__ == "__main__":
    print("🤖 SPOT ROBOT EMOTION RECOGNITION TRAINING")
    print("=" * 60)
    
    # Train the model
    model, losses, accuracies, test_acc = train_model()
    
    # Test on single video
    test_single_video()
    
    print("\n🎉 Training complete!")
    print("🚀 Ready to deploy to Spot robot!")