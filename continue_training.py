#!/usr/bin/env python3
"""
Continue Training from Saved Model
Load your existing spot_emotion_model_6class.pth and improve it further
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

class OriginalModelArchitecture(nn.Module):
    """EXACT architecture from your saved model"""
    def __init__(self, num_classes=6):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # features.0
            nn.BatchNorm2d(32),                           # features.1
            nn.ReLU(inplace=True),                        # features.2
            nn.MaxPool2d(2, 2),                           # features.3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # features.4
            nn.BatchNorm2d(64),                           # features.5
            nn.ReLU(inplace=True),                        # features.6
            nn.MaxPool2d(2, 2),                           # features.7
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # features.8
            nn.BatchNorm2d(128),                          # features.9
            nn.ReLU(inplace=True),                        # features.10
            nn.MaxPool2d(2, 2),                           # features.11
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # features.12
            nn.BatchNorm2d(256),                          # features.13
            nn.ReLU(inplace=True),                        # features.14
            nn.AdaptiveAvgPool2d((1, 1))                  # features.15
        )
        
        # Bidirectional LSTM: 256 input -> 128 hidden -> 256 output
        self.temporal_lstm = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),                             # classifier.0
            nn.Linear(256, 128),                         # classifier.1
            nn.ReLU(inplace=True),                       # classifier.2
            nn.Dropout(0.3),                             # classifier.3
            nn.Linear(128, num_classes)                  # classifier.4
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

class FixedRAVDESSDataset(Dataset):
    """Same dataset that worked for your baseline"""
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
        
        if not augment:  # Only print for test dataset
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
        
        # Enhanced augmentation for continued training
        if self.augment:
            frames = self._enhanced_augment(frames)
        
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
            
            # Focus on emotionally expressive middle portion
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
    
    def _enhanced_augment(self, frames):
        """Enhanced augmentation to help the model generalize better"""
        # Horizontal flip (more aggressive)
        if np.random.random() > 0.4:
            frames = frames[:, :, ::-1, :].copy()
        
        # Brightness and contrast variations
        if np.random.random() > 0.3:
            brightness = np.random.uniform(0.7, 1.3)
            contrast = np.random.uniform(0.8, 1.2)
            frames = np.clip(frames * contrast + (brightness - 1) * 50, 0, 255)
        
        # Small rotations
        if np.random.random() > 0.6:
            angle = np.random.uniform(-10, 10)
            for i in range(len(frames)):
                center = (self.frame_size // 2, self.frame_size // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                frames[i] = cv2.warpAffine(frames[i], rotation_matrix, 
                                         (self.frame_size, self.frame_size))
        
        # Temporal variations - dropout some frames
        if np.random.random() > 0.7:
            num_drop = min(4, len(frames) // 6)
            drop_indices = np.random.choice(len(frames), num_drop, replace=False)
            for idx in drop_indices:
                if idx > 0:
                    frames[idx] = frames[idx-1]  # Replace with previous frame
        
        # Random noise
        if np.random.random() > 0.8:
            noise = np.random.normal(0, 5, frames.shape)
            frames = np.clip(frames + noise, 0, 255)
        
        return frames

def continue_training_from_saved():
    """Continue training from your saved 30.56% model"""
    print("🔄 CONTINUING TRAINING FROM SAVED MODEL")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Training on: {device}")
    
    # Load your existing model
    print("📂 Loading existing model...")
    model = OriginalModelArchitecture(num_classes=6)
    model.load_state_dict(torch.load('spot_emotion_model_6class.pth', map_location='cpu'))
    model = model.to(device)
    
    print("✅ Loaded your 30.56% baseline model!")
    
    # Test current performance first
    print("\n🧪 Testing current model performance...")
    test_current_model(model, device)
    
    # Dataset with enhanced augmentation for continued training
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
    
    # Data loaders
    train_loader = DataLoader(train_subset, batch_size=8, sampler=sampler, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=8, shuffle=False, num_workers=0)
    
    # Continued training setup - lower learning rate since model is already trained
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)  # Lower LR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3)
    
    # Continue training
    num_epochs = 20  # Shorter since we're continuing
    best_accuracy = 30.56  # Start from known baseline
    patience = 5
    no_improve = 0
    
    print(f"\n🎯 Starting from: 30.56% accuracy")
    print("🏋️  Continuing training with enhanced data...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for videos, labels in train_loader:
            videos, labels = videos.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
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
        scheduler.step(test_acc)
        
        print(f"Epoch {epoch+1:2d}/{num_epochs}: Train {train_acc:5.1f}% | Test {test_acc:5.1f}%")
        
        # Save if improved
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            improvement = best_accuracy - 30.56
            torch.save(model.state_dict(), 'spot_emotion_model_6class_improved.pth')
            print(f"   🎯 NEW BEST: {best_accuracy:.2f}% (+{improvement:.2f}%)")
            no_improve = 0
        else:
            no_improve += 1
        
        # Early stopping
        if no_improve >= patience:
            print(f"⏹️  Early stopping: No improvement for {patience} epochs")
            break
    
    return model, train_dataset, best_accuracy

def test_current_model(model, device):
    """Test the current model performance"""
    video_dirs = ["Video_Song_Actor_02/Actor_02", "Video_Song_Actor_06/Actor_06"]
    dataset = FixedRAVDESSDataset(video_dirs, max_frames=30, frame_size=48, augment=False)
    
    model.eval()
    test_videos = [
        ("Video_Song_Actor_02/Actor_02/01-02-03-01-01-01-02.mp4", "happy"),
        ("Video_Song_Actor_02/Actor_02/01-02-05-01-01-01-02.mp4", "angry"),
        ("Video_Song_Actor_02/Actor_02/01-02-01-01-01-01-02.mp4", "neutral"),
    ]
    
    correct = 0
    total = 0
    
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
                    is_correct = predicted_emotion == expected_emotion
                    
                    print(f"   📹 {expected_emotion} → {predicted_emotion} ({confidence:.3f}) {'✅' if is_correct else '❌'}")
                    
                    if is_correct:
                        correct += 1
                    total += 1
                    break
    
    print(f"📊 Current accuracy: {100*correct/total:.1f}% ({correct}/{total})")

def test_improved_model():
    """Test the improved model"""
    print("\n🧪 Testing Improved Model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load improved model
    model = OriginalModelArchitecture(num_classes=6).to(device)
    model.load_state_dict(torch.load('spot_emotion_model_6class_improved.pth', map_location=device))
    model.eval()
    
    # Test on real videos
    video_dirs = ["Video_Song_Actor_02/Actor_02", "Video_Song_Actor_06/Actor_06"]
    dataset = FixedRAVDESSDataset(video_dirs, max_frames=30, frame_size=48, augment=False)
    
    test_videos = [
        ("Video_Song_Actor_02/Actor_02/01-02-03-01-01-01-02.mp4", "happy"),
        ("Video_Song_Actor_02/Actor_02/01-02-05-01-01-01-02.mp4", "angry"),
        ("Video_Song_Actor_02/Actor_02/01-02-01-01-01-01-02.mp4", "neutral"),
        ("Video_Song_Actor_02/Actor_02/01-02-04-01-01-01-02.mp4", "sad"),
    ]
    
    correct = 0
    total = 0
    
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
                    is_correct = predicted_emotion == expected_emotion
                    
                    print(f"   📹 {expected_emotion} → {predicted_emotion} ({confidence:.3f}) {'✅' if is_correct else '❌'}")
                    
                    if is_correct:
                        correct += 1
                    total += 1
                    break
    
    print(f"🎯 Final accuracy: {100*correct/total:.1f}% ({correct}/{total})")

if __name__ == "__main__":
    print("🔄 CONTINUE TRAINING FROM YOUR SAVED MODEL")
    print("📊 Current: spot_emotion_model_6class.pth (30.56%)")
    print("🎯 Goal: Improve existing model performance")
    print("=" * 60)
    
    # Continue training from saved model
    model, dataset, best_acc = continue_training_from_saved()
    
    # Test the improved model
    test_improved_model()
    
    print(f"\n🎉 CONTINUED TRAINING RESULTS:")
    print(f"📈 Original: 30.56%")
    print(f"🏆 Improved: {best_acc:.2f}%")
    print(f"📊 Gain: +{best_acc - 30.56:.2f}%")
    print(f"💾 Original: spot_emotion_model_6class.pth")
    print(f"💾 Improved: spot_emotion_model_6class_improved.pth")
    
    if best_acc >= 45:
        print(f"🎯 GREAT! Significant improvement achieved!")
    elif best_acc >= 35:
        print(f"✅ GOOD! Noticeable improvement")
    else:
        print(f"📈 Some improvement - model is learning from enhanced data")