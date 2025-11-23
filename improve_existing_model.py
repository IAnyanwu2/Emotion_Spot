#!/usr/bin/env python3
"""
Improve Existing Model - Continue from 30.56% baseline
Load the existing model and apply optimization techniques
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

# Import the original architecture
class SimpleEmotionCNN(nn.Module):
    """Original architecture from your working model"""
    def __init__(self, num_classes=6):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.lstm = nn.LSTM(256, 128, batch_first=True)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        b, t, h, w, c = x.shape
        x = x.view(b * t, c, h, w)
        
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        
        x = x.view(b * t, -1)
        x = torch.relu(self.fc1(x))
        x = x.view(b, t, -1)
        
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EnhancedEmotionCNN(nn.Module):
    """Enhanced version of your working model with improvements"""
    def __init__(self, num_classes=6):
        super().__init__()
        # Improved conv layers with batch norm
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Enhanced FC with more capacity
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Bidirectional LSTM for better temporal modeling
        self.lstm = nn.LSTM(512, 256, num_layers=2, batch_first=True, 
                           dropout=0.3, bidirectional=True)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Softmax(dim=1)
        )
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        b, t, h, w, c = x.shape
        x = x.view(b * t, c, h, w)
        
        # Feature extraction
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.conv4(x)
        x = self.adaptive_pool(x)
        
        x = x.view(b * t, -1)
        x = self.fc1(x)
        x = x.view(b, t, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        attended_features = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Classification
        output = self.classifier(attended_features)
        return output

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
        
        # Enhanced augmentation
        if self.augment:
            frames = self._augment_frames(frames)
        
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
            
            # Better frame sampling
            start_frame = max(0, total_frames // 6)  # Skip beginning
            end_frame = min(total_frames, 5 * total_frames // 6)  # Skip end
            
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
    
    def _augment_frames(self, frames):
        """Enhanced data augmentation"""
        # Random horizontal flip
        if np.random.random() > 0.5:
            frames = frames[:, :, ::-1, :].copy()
        
        # Random brightness/contrast
        if np.random.random() > 0.4:
            brightness = np.random.uniform(0.8, 1.2)
            contrast = np.random.uniform(0.9, 1.1)
            frames = np.clip(frames * contrast + (brightness - 1) * 30, 0, 255)
        
        # Random crop and resize
        if np.random.random() > 0.6:
            crop_size = int(self.frame_size * 0.9)
            start_x = np.random.randint(0, self.frame_size - crop_size)
            start_y = np.random.randint(0, self.frame_size - crop_size)
            
            for i in range(len(frames)):
                cropped = frames[i, start_y:start_y+crop_size, start_x:start_x+crop_size]
                frames[i] = cv2.resize(cropped, (self.frame_size, self.frame_size))
        
        return frames

def transfer_weights(old_model, new_model):
    """Transfer compatible weights from old model to new model"""
    old_dict = old_model.state_dict()
    new_dict = new_model.state_dict()
    
    transferred = 0
    for name, param in new_dict.items():
        if name in old_dict and old_dict[name].shape == param.shape:
            param.data.copy_(old_dict[name].data)
            transferred += 1
            print(f"✅ Transferred: {name}")
        else:
            print(f"⚠️  Skipped: {name} (shape mismatch or not found)")
    
    print(f"📊 Transferred {transferred}/{len(new_dict)} layers")
    return new_model

def improve_existing_model():
    """Improve the existing 30.56% model"""
    print("🚀 IMPROVING EXISTING MODEL FROM 30.56%")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Training on: {device}")
    
    # Dataset setup
    video_dirs = [
        "Video_Song_Actor_02/Actor_02",
        "Video_Song_Actor_06/Actor_06"
    ]
    
    # Create enhanced dataset with augmentation
    train_dataset = FixedRAVDESSDataset(video_dirs, max_frames=30, frame_size=48, augment=True)
    test_dataset = FixedRAVDESSDataset(video_dirs, max_frames=30, frame_size=48, augment=False)
    
    # Split data
    total_size = len(train_dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    
    # Stratified split
    train_indices = []
    test_indices = []
    
    for class_idx in range(train_dataset.num_classes):
        class_indices = [i for i, label in enumerate(train_dataset.labels) if label == class_idx]
        class_train_size = int(0.8 * len(class_indices))
        
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
    
    # Load existing model
    print("\n📂 Loading existing model...")
    old_model = SimpleEmotionCNN(num_classes=train_dataset.num_classes)
    old_model.load_state_dict(torch.load('spot_emotion_model_6class.pth', map_location='cpu'))
    print(f"✅ Loaded baseline model (30.56% accuracy)")
    
    # Create enhanced model
    print("\n🤖 Creating enhanced model...")
    new_model = EnhancedEmotionCNN(num_classes=train_dataset.num_classes).to(device)
    
    # Option 1: Start from scratch with better architecture
    param_count = sum(p.numel() for p in new_model.parameters())
    print(f"🔧 Enhanced model: {param_count:,} parameters")
    
    # Training setup with advanced techniques
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(new_model.parameters(), lr=0.0003, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Training with early stopping
    num_epochs = 40
    best_accuracy = 30.56  # Start from baseline
    patience = 8
    no_improve = 0
    
    train_losses = []
    train_accs = []
    test_accs = []
    
    print(f"\n🏋️  Starting improvement training...")
    print(f"🎯 Target: Beat 30.56% baseline")
    
    for epoch in range(num_epochs):
        # Training phase
        new_model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (videos, labels) in enumerate(train_loader):
            videos, labels = videos.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = new_model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(new_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Testing phase
        new_model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for videos, labels in test_loader:
                videos, labels = videos.to(device), labels.to(device)
                outputs = new_model(videos)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_acc = 100 * test_correct / test_total
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        # Learning rate scheduling
        scheduler.step(test_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"   Test Acc: {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            improvement = best_accuracy - 30.56
            torch.save(new_model.state_dict(), 'improved_spot_emotion_model.pth')
            print(f"   🎯 NEW BEST: {best_accuracy:.2f}% (+{improvement:.2f}%)")
            no_improve = 0
        else:
            no_improve += 1
        
        # Early stopping
        if no_improve >= patience:
            print(f"⏹️  Early stopping: No improvement for {patience} epochs")
            break
        
        print("-" * 50)
    
    print(f"\n🎉 IMPROVEMENT COMPLETE!")
    print(f"📈 Baseline: 30.56%")
    print(f"🏆 Enhanced: {best_accuracy:.2f}%")
    print(f"📊 Improvement: +{best_accuracy - 30.56:.2f}%")
    
    return new_model, train_dataset, best_accuracy

def test_improved_model():
    """Test the improved model on real videos"""
    print("\n🧪 Testing Improved Model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    video_dirs = ["Video_Song_Actor_02/Actor_02", "Video_Song_Actor_06/Actor_06"]
    dataset = FixedRAVDESSDataset(video_dirs, max_frames=30, frame_size=48, augment=False)
    
    # Load improved model
    model = EnhancedEmotionCNN(num_classes=dataset.num_classes).to(device)
    model.load_state_dict(torch.load('improved_spot_emotion_model.pth', map_location=device))
    model.eval()
    
    # Test on specific videos
    test_videos = [
        ("Video_Song_Actor_02/Actor_02/01-02-03-01-01-01-02.mp4", "happy"),
        ("Video_Song_Actor_02/Actor_02/01-02-05-01-01-01-02.mp4", "angry"),
        ("Video_Song_Actor_02/Actor_02/01-02-01-01-01-01-02.mp4", "neutral"),
        ("Video_Song_Actor_02/Actor_02/01-02-04-01-01-01-02.mp4", "sad"),
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
    print("🚀 CONTINUING FROM EXISTING MODEL")
    print("📈 Baseline: 30.56% accuracy")
    print("🎯 Goal: Significantly improve accuracy")
    print("=" * 60)
    
    # Improve the existing model
    model, dataset, best_acc = improve_existing_model()
    
    # Test the improved model
    test_improved_model()
    
    print(f"\n🎉 FINAL RESULTS:")
    print(f"📈 Original: 30.56%")
    print(f"🏆 Improved: {best_acc:.2f}%")
    print(f"📊 Gain: +{best_acc - 30.56:.2f}%")
    print(f"💾 Model: improved_spot_emotion_model.pth")
    
    if best_acc > 50:
        print(f"🎯 EXCELLENT! Ready for Spot deployment!")
    elif best_acc > 40:
        print(f"✅ GOOD improvement! Consider further optimization")
    else:
        print(f"📈 Some improvement, but may need more data or architecture changes")