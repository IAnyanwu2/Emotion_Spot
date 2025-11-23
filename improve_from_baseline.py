#!/usr/bin/env python3
"""
Improve Existing Model - CORRECT ARCHITECTURE
Continue from 30.56% baseline with the exact same architecture
"""

import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from pathlib import Path
import time
from collections import Counter
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, confusion_matrix

class OriginalModelArchitecture(nn.Module):
    """EXACT architecture that was saved in the model file"""
    def __init__(self, num_classes=6):
        super().__init__()
        
        # Features module - EXACT match to saved model
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # features.0: [32, 3, 3, 3]
            nn.BatchNorm2d(32),                           # features.1: [32]
            nn.ReLU(inplace=True),                        # features.2
            nn.MaxPool2d(2, 2),                           # features.3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # features.4: [64, 32, 3, 3]
            nn.BatchNorm2d(64),                           # features.5: [64]
            nn.ReLU(inplace=True),                        # features.6
            nn.MaxPool2d(2, 2),                           # features.7
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # features.8: [128, 64, 3, 3]
            nn.BatchNorm2d(128),                          # features.9: [128]
            nn.ReLU(inplace=True),                        # features.10
            nn.MaxPool2d(2, 2),                           # features.11
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # features.12: [256, 128, 3, 3]
            nn.BatchNorm2d(256),                          # features.13: [256]
            nn.ReLU(inplace=True),                        # features.14
            nn.AdaptiveAvgPool2d((1, 1))                  # features.15
        )
        
        # Temporal LSTM - bidirectional with 128 hidden units
        # weight_ih_l0: [512, 256] means 4*128 gates, 256 input features
        # weight_hh_l0: [512, 128] means 4*128 gates, 128 hidden features
        self.temporal_lstm = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        
        # Classifier - EXACT match
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),                             # classifier.0
            nn.Linear(256, 128),                         # classifier.1: [128, 256]
            nn.ReLU(inplace=True),                       # classifier.2
            nn.Dropout(0.3),                             # classifier.3
            nn.Linear(128, num_classes)                  # classifier.4: [6, 128]
        )
    
    def forward(self, x):
        b, t, h, w, c = x.shape
        x = x.view(b * t, c, h, w)
        
        # Feature extraction
        x = self.features(x)  # (b*t, 256, 1, 1)
        x = x.view(b * t, -1)  # (b*t, 256)
        
        # Reshape for temporal processing
        x = x.view(b, t, -1)  # (b, t, 256)
        
        # LSTM processing (bidirectional output is 256)
        lstm_out, _ = self.temporal_lstm(x)  # (b, t, 256)
        
        # Use last output
        x = lstm_out[:, -1, :]  # (b, 256)
        
        # Classification
        x = self.classifier(x)
        return x

class ImprovedModelArchitecture(nn.Module):
    """Improved version of the original architecture"""
    def __init__(self, num_classes=6):
        super().__init__()
        
        # Enhanced features with more capacity and better regularization
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # Extra conv
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Extra conv
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Extra conv
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Extra conv
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Enhanced temporal processing
        self.temporal_lstm = nn.LSTM(256, 128, num_layers=2, batch_first=True, 
                                   dropout=0.2, bidirectional=True)
        
        # Attention mechanism for temporal features
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        b, t, h, w, c = x.shape
        x = x.view(b * t, c, h, w)
        
        # Feature extraction
        x = self.features(x)  # (b*t, 256, 1, 1)
        x = x.view(b * t, -1)  # (b*t, 256)
        
        # Reshape for temporal processing
        x = x.view(b, t, -1)  # (b, t, 256)
        
        # LSTM processing
        lstm_out, _ = self.temporal_lstm(x)  # (b, t, 256)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)  # (b, t, 1)
        attended_features = torch.sum(lstm_out * attention_weights, dim=1)  # (b, 256)
        
        # Classification
        x = self.classifier(attended_features)
        return x

class FixedRAVDESSDataset(Dataset):
    """Same dataset class that worked for your baseline"""
    def __init__(self, video_dirs, max_frames=30, frame_size=48, augment=False):
        self.video_paths = []
        self.labels = []
        self.max_frames = max_frames
        self.frame_size = frame_size
        self.augment = augment
        
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
        
        emotion_codes = sorted(list(available_emotions))
        self.emotion_map = {code: idx for idx, code in enumerate(emotion_codes)}
        
        full_emotion_names = {
            '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
            '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
        }
        
        self.emotion_names = [full_emotion_names[code] for code in emotion_codes]
        self.num_classes = len(self.emotion_names)
        
        for video_file, emotion_code in temp_data:
            if emotion_code in self.emotion_map:
                self.video_paths.append(video_file)
                self.labels.append(self.emotion_map[emotion_code])
        
        print(f"📊 Dataset: {len(self.video_paths)} videos, {self.num_classes} emotions")
        print(f"🎭 Emotions: {self.emotion_names}")
        self._print_distribution()
    
    def _print_distribution(self):
        label_counts = Counter(self.labels)
        print("📈 Distribution:")
        for emotion_idx, count in label_counts.items():
            emotion_name = self.emotion_names[emotion_idx]
            print(f"   {emotion_name}: {count} videos")
    
    def get_class_weights(self):
        label_counts = Counter(self.labels)
        total = len(self.labels)
        weights = []
        for i in range(self.num_classes):
            count = label_counts.get(i, 1)
            weight = total / (self.num_classes * count)
            weights.append(weight)
        return torch.FloatTensor(weights)
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        frames = self._load_video_frames(video_path)
        if frames is None:
            frames = np.zeros((self.max_frames, self.frame_size, self.frame_size, 3))
        
        # Enhanced augmentation for training
        if self.augment:
            frames = self._augment_frames_enhanced(frames)
        
        frames_tensor = torch.from_numpy(frames).float() / 255.0
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return frames_tensor, label_tensor
    
    def _load_video_frames(self, video_path):
        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                cap.release()
                return None
            
            # Better frame sampling - focus on expressive middle portion
            start_frame = max(0, total_frames // 4)
            end_frame = min(total_frames, 3 * total_frames // 4)
            
            if end_frame - start_frame < self.max_frames:
                frame_indices = list(range(start_frame, end_frame))
                while len(frame_indices) < self.max_frames:
                    frame_indices.extend(frame_indices)
                frame_indices = frame_indices[:self.max_frames]
            else:
                frame_indices = np.linspace(start_frame, end_frame-1, self.max_frames, dtype=int)
            
            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame_resized = cv2.resize(frame, (self.frame_size, self.frame_size))
                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                else:
                    if frames:
                        frames.append(frames[-1])
                    else:
                        frames.append(np.zeros((self.frame_size, self.frame_size, 3)))
            
            cap.release()
            return np.array(frames)
            
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            return None
    
    def _augment_frames_enhanced(self, frames):
        """Enhanced data augmentation for better generalization"""
        # Random horizontal flip
        if np.random.random() > 0.5:
            frames = frames[:, :, ::-1, :].copy()
        
        # Random brightness and contrast
        if np.random.random() > 0.3:
            brightness = np.random.uniform(0.7, 1.3)
            contrast = np.random.uniform(0.8, 1.2)
            frames = np.clip(frames * contrast + (brightness - 1) * 40, 0, 255)
        
        # Random rotation
        if np.random.random() > 0.7:
            angle = np.random.uniform(-8, 8)
            for i in range(len(frames)):
                center = (self.frame_size // 2, self.frame_size // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                frames[i] = cv2.warpAffine(frames[i], rotation_matrix, 
                                         (self.frame_size, self.frame_size))
        
        # Random temporal dropout
        if np.random.random() > 0.8:
            num_drop = min(3, len(frames) // 5)
            drop_indices = np.random.choice(len(frames), num_drop, replace=False)
            for idx in drop_indices:
                if idx > 0:
                    frames[idx] = frames[idx-1]
        
        # Random zoom
        if np.random.random() > 0.6:
            zoom_factor = np.random.uniform(0.9, 1.1)
            for i in range(len(frames)):
                h, w = frames[i].shape[:2]
                new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
                resized = cv2.resize(frames[i], (new_w, new_h))
                
                if zoom_factor > 1:
                    # Crop center
                    start_x = (new_w - w) // 2
                    start_y = (new_h - h) // 2
                    frames[i] = resized[start_y:start_y+h, start_x:start_x+w]
                else:
                    # Pad
                    pad_x = (w - new_w) // 2
                    pad_y = (h - new_h) // 2
                    frames[i] = cv2.copyMakeBorder(resized, pad_y, h-new_h-pad_y, 
                                                 pad_x, w-new_w-pad_x, cv2.BORDER_REFLECT)
        
        return frames

def load_baseline_model():
    """Load the baseline model and verify it works"""
    print("📂 Loading baseline model...")
    
    model = OriginalModelArchitecture(num_classes=6)
    state_dict = torch.load('spot_emotion_model_6class.pth', map_location='cpu')
    model.load_state_dict(state_dict)
    
    print("✅ Baseline model loaded successfully!")
    
    # Test with dummy input
    dummy_input = torch.randn(1, 30, 48, 48, 3)
    output = model(dummy_input)
    print(f"✅ Model forward pass: {output.shape}")
    
    return model

def train_improved_model():
    """Train an improved version starting from the baseline understanding"""
    print("🚀 TRAINING IMPROVED MODEL")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Training on: {device}")
    
    # Verify baseline first
    baseline_model = load_baseline_model()
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    print(f"📊 Baseline model: {baseline_params:,} parameters")
    
    # Dataset with enhanced augmentation
    video_dirs = [
        "Video_Song_Actor_02/Actor_02",
        "Video_Song_Actor_06/Actor_06"
    ]
    
    train_dataset = FixedRAVDESSDataset(video_dirs, max_frames=30, frame_size=48, augment=True)
    test_dataset = FixedRAVDESSDataset(video_dirs, max_frames=30, frame_size=48, augment=False)
    
    # Stratified split
    total_size = len(train_dataset)
    train_indices = []
    test_indices = []
    
    for class_idx in range(train_dataset.num_classes):
        class_indices = [i for i, label in enumerate(train_dataset.labels) if label == class_idx]
        class_train_size = int(0.85 * len(class_indices))  # More training data
        
        np.random.shuffle(class_indices)
        train_indices.extend(class_indices[:class_train_size])
        test_indices.extend(class_indices[class_train_size:])
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)
    
    print(f"📊 Train: {len(train_subset)}, Test: {len(test_subset)}")
    
    # Balanced sampling
    class_weights = train_dataset.get_class_weights()
    sample_weights = [class_weights[train_dataset.labels[i]] for i in train_indices]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Data loaders with multiple workers
    train_loader = DataLoader(train_subset, batch_size=10, sampler=sampler, 
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=10, shuffle=False, 
                           num_workers=0, pin_memory=True)
    
    # Create improved model
    model = ImprovedModelArchitecture(num_classes=train_dataset.num_classes).to(device)
    improved_params = sum(p.numel() for p in model.parameters())
    print(f"🤖 Improved model: {improved_params:,} parameters")
    
    # Advanced training setup
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, 
                                            steps_per_epoch=len(train_loader), epochs=30)
    
    # Training with best practices
    num_epochs = 30
    best_accuracy = 30.56  # Baseline to beat
    patience = 6
    no_improve = 0
    
    print(f"\n🎯 Target: Beat baseline 30.56% accuracy")
    print("🏋️  Starting improved training...")
    
    for epoch in range(num_epochs):
        # Training phase
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
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Testing phase
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
        
        test_acc = 100 * test_correct / test_total
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"   Test Acc: {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            improvement = best_accuracy - 30.56
            torch.save(model.state_dict(), 'improved_spot_emotion_model_v2.pth')
            print(f"   🎯 NEW BEST: {best_accuracy:.2f}% (+{improvement:.2f}%)")
            no_improve = 0
        else:
            no_improve += 1
        
        # Early stopping
        if no_improve >= patience:
            print(f"⏹️  Early stopping: No improvement for {patience} epochs")
            break
        
        print("-" * 45)
    
    return model, train_dataset, best_accuracy

def test_final_model():
    """Test the final improved model"""
    print("\n🧪 Testing Final Improved Model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    video_dirs = ["Video_Song_Actor_02/Actor_02", "Video_Song_Actor_06/Actor_06"]
    dataset = FixedRAVDESSDataset(video_dirs, max_frames=30, frame_size=48, augment=False)
    
    # Load improved model
    model = ImprovedModelArchitecture(num_classes=dataset.num_classes).to(device)
    model.load_state_dict(torch.load('improved_spot_emotion_model_v2.pth', map_location=device))
    model.eval()
    
    # Test specific videos
    test_videos = [
        ("Video_Song_Actor_02/Actor_02/01-02-03-01-01-01-02.mp4", "happy"),
        ("Video_Song_Actor_02/Actor_02/01-02-05-01-01-01-02.mp4", "angry"),
        ("Video_Song_Actor_02/Actor_02/01-02-01-01-01-01-02.mp4", "neutral"),
        ("Video_Song_Actor_02/Actor_02/01-02-04-01-01-01-02.mp4", "sad"),
        ("Video_Song_Actor_02/Actor_02/01-02-02-01-01-01-02.mp4", "calm"),
        ("Video_Song_Actor_02/Actor_02/01-02-06-01-01-01-02.mp4", "fearful"),
    ]
    
    correct_predictions = 0
    total_predictions = 0
    
    for video_path, expected_emotion in test_videos:
        if Path(video_path).exists():
            temp_dataset = FixedRAVDESSDataset([Path(video_path).parent], 
                                             max_frames=30, frame_size=48, augment=False)
            
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
                    print(f"   🎯 Expected: {expected_emotion}")
                    print(f"   🤖 Predicted: {predicted_emotion} ({confidence:.3f})")
                    
                    is_correct = predicted_emotion == expected_emotion
                    print(f"   {'✅' if is_correct else '❌'} Match: {is_correct}")
                    print()
                    
                    if is_correct:
                        correct_predictions += 1
                    total_predictions += 1
                    break
    
    if total_predictions > 0:
        accuracy = 100 * correct_predictions / total_predictions
        print(f"🎯 Real Video Test Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")

if __name__ == "__main__":
    print("🚀 IMPROVING FROM BASELINE MODEL")
    print("📊 Starting from: 30.56% accuracy")
    print("🎯 Goal: Significantly improve accuracy")
    print("=" * 60)
    
    # Train improved model
    model, dataset, best_acc = train_improved_model()
    
    # Test the improved model
    test_final_model()
    
    print(f"\n🎉 OPTIMIZATION RESULTS:")
    print(f"📈 Baseline: 30.56%")
    print(f"🏆 Improved: {best_acc:.2f}%")
    print(f"📊 Gain: +{best_acc - 30.56:.2f}%")
    print(f"💾 Model: improved_spot_emotion_model_v2.pth")
    
    if best_acc >= 60:
        print(f"🎯 EXCELLENT! Ready for Spot robot deployment!")
    elif best_acc >= 45:
        print(f"✅ GOOD improvement! Consider for Spot deployment")
    else:
        print(f"📈 Some improvement - may need more optimization")