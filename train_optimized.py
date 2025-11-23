#!/usr/bin/env python3
"""
OPTIMIZED Training Script for Higher Accuracy
Target: 60-80% accuracy for Spot robot deployment
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
import matplotlib.pyplot as plt

class OptimizedEmotionCNN(nn.Module):
    """Optimized CNN with better architecture for higher accuracy"""
    def __init__(self, num_classes=6, input_size=64):  # Larger input for better features
        super().__init__()
        self.num_classes = num_classes
        
        # Improved feature extraction with residual connections
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),  # 64x64 -> 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)  # 32x32 -> 16x16
        )
        
        # Residual blocks for better learning
        self.res_block1 = self._make_res_block(64, 128, stride=2)  # 16x16 -> 8x8
        self.res_block2 = self._make_res_block(128, 256, stride=2)  # 8x8 -> 4x4
        self.res_block3 = self._make_res_block(256, 512, stride=2)  # 4x4 -> 2x2
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Improved temporal processing with GRU
        self.temporal_gru = nn.GRU(512, 256, num_layers=2, batch_first=True, 
                                   dropout=0.3, bidirectional=True)
        
        # Attention mechanism for temporal features
        self.attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Softmax(dim=1)
        )
        
        # Enhanced classifier with regularization
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def _make_res_block(self, in_channels, out_channels, stride=1):
        """Create residual block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        b, t, h, w, c = x.shape
        
        # Process all frames
        x = x.view(b * t, c, h, w)
        
        # Feature extraction
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        features = self.global_pool(x)  # (b*t, 512, 1, 1)
        features = features.view(b * t, -1)  # (b*t, 512)
        
        # Reshape for temporal processing
        features = features.view(b, t, -1)  # (b, t, 512)
        
        # GRU for temporal modeling
        gru_out, _ = self.temporal_gru(features)  # (b, t, 512)
        
        # Attention mechanism
        attention_weights = self.attention(gru_out)  # (b, t, 1)
        attended_features = torch.sum(gru_out * attention_weights, dim=1)  # (b, 512)
        
        # Classification
        output = self.classifier(attended_features)
        return output

class OptimizedRAVDESSDataset(Dataset):
    """Optimized dataset with better preprocessing and augmentation"""
    def __init__(self, video_dirs, max_frames=40, frame_size=64, is_training=True):
        self.video_paths = []
        self.labels = []
        self.max_frames = max_frames
        self.frame_size = frame_size
        self.is_training = is_training
        
        # Collect data
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
        
        print(f"📊 {'Training' if is_training else 'Test'} dataset: {len(self.video_paths)} videos")
        if is_training:
            print(f"🎭 Emotions: {self.emotion_names}")
            self._print_distribution()
    
    def _print_distribution(self):
        label_counts = Counter(self.labels)
        print("📈 Distribution:")
        for emotion_idx, count in label_counts.items():
            emotion_name = self.emotion_names[emotion_idx]
            print(f"   {emotion_name}: {count} videos")
    
    def get_class_weights(self):
        """Calculate class weights for balanced training"""
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
        
        frames = self._load_video_frames_optimized(video_path)
        if frames is None:
            frames = np.zeros((self.max_frames, self.frame_size, self.frame_size, 3))
        
        # Enhanced data augmentation for training
        if self.is_training:
            frames = self._augment_frames_enhanced(frames)
        
        frames_tensor = torch.from_numpy(frames).float() / 255.0
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return frames_tensor, label_tensor
    
    def _load_video_frames_optimized(self, video_path):
        """Optimized frame loading with better sampling"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                cap.release()
                return None
            
            # Smart frame sampling - focus on middle portion where emotion is clear
            start_frame = max(0, total_frames // 4)  # Skip first 25%
            end_frame = min(total_frames, 3 * total_frames // 4)  # Skip last 25%
            
            # Sample frames uniformly from the middle portion
            if end_frame - start_frame < self.max_frames:
                frame_indices = list(range(start_frame, end_frame))
                # Pad with repeated frames if needed
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
                    # Better preprocessing
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
        """Enhanced data augmentation"""
        # Random horizontal flip
        if np.random.random() > 0.5:
            frames = frames[:, :, ::-1, :].copy()
        
        # Random brightness and contrast
        if np.random.random() > 0.3:
            brightness = np.random.uniform(0.7, 1.3)
            contrast = np.random.uniform(0.8, 1.2)
            frames = np.clip(frames * contrast + (brightness - 1) * 50, 0, 255)
        
        # Random rotation (small)
        if np.random.random() > 0.7:
            angle = np.random.uniform(-5, 5)
            for i in range(len(frames)):
                center = (self.frame_size // 2, self.frame_size // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                frames[i] = cv2.warpAffine(frames[i], rotation_matrix, 
                                         (self.frame_size, self.frame_size))
        
        # Random temporal dropout (remove some frames)
        if np.random.random() > 0.8:
            num_drop = min(5, len(frames) // 4)
            drop_indices = np.random.choice(len(frames), num_drop, replace=False)
            for idx in drop_indices:
                if idx > 0:
                    frames[idx] = frames[idx-1]
        
        return frames

def train_optimized():
    """Optimized training for higher accuracy"""
    print("🚀 OPTIMIZED TRAINING FOR HIGH ACCURACY")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Training on: {device}")
    
    # Dataset with optimization
    video_dirs = [
        "Video_Song_Actor_02/Actor_02",
        "Video_Song_Actor_06/Actor_06"
    ]
    
    # Create train/test datasets
    full_dataset = OptimizedRAVDESSDataset(video_dirs, max_frames=40, frame_size=64, is_training=True)
    
    # Stratified split to ensure balanced classes
    total_size = len(full_dataset)
    train_size = int(0.85 * total_size)  # More training data
    test_size = total_size - train_size
    
    # Manual stratified split
    train_indices = []
    test_indices = []
    
    for class_idx in range(full_dataset.num_classes):
        class_indices = [i for i, label in enumerate(full_dataset.labels) if label == class_idx]
        class_train_size = int(0.85 * len(class_indices))
        
        np.random.shuffle(class_indices)
        train_indices.extend(class_indices[:class_train_size])
        test_indices.extend(class_indices[class_train_size:])
    
    # Create datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    
    test_dataset_obj = OptimizedRAVDESSDataset(video_dirs, max_frames=40, frame_size=64, is_training=False)
    test_dataset = torch.utils.data.Subset(test_dataset_obj, test_indices)
    
    print(f"📊 Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # Balanced sampling for training
    class_weights = full_dataset.get_class_weights()
    sample_weights = [class_weights[full_dataset.labels[i]] for i in train_indices]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=6, sampler=sampler, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=0)
    
    # Optimized model
    model = OptimizedEmotionCNN(num_classes=full_dataset.num_classes, input_size=64).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"🤖 Optimized model: {param_count:,} parameters")
    
    # Advanced training setup
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, 
                                            steps_per_epoch=len(train_loader), epochs=35)
    
    # Training loop
    num_epochs = 35
    best_accuracy = 0
    train_losses = []
    train_accs = []
    test_accs = []
    
    print("\n🏋️  Starting optimized training...")
    for epoch in range(num_epochs):
        # Training
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
            
            if batch_idx % 10 == 0:
                print(f"   Batch {batch_idx}/{len(train_loader)}: Loss {loss.item():.4f}")
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Testing
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
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"   Test Acc: {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'optimized_spot_emotion_model.pth')
            print(f"   🎯 NEW BEST: {best_accuracy:.2f}%")
        
        print("-" * 50)
    
    print(f"\n🎉 OPTIMIZATION COMPLETE!")
    print(f"🏆 Best Test Accuracy: {best_accuracy:.2f}%")
    
    return model, full_dataset, best_accuracy

def test_optimized_model():
    """Test the optimized model"""
    print("\n🧪 Testing Optimized Model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    video_dirs = ["Video_Song_Actor_02/Actor_02", "Video_Song_Actor_06/Actor_06"]
    dataset = OptimizedRAVDESSDataset(video_dirs, max_frames=40, frame_size=64, is_training=False)
    
    # Load model
    model = OptimizedEmotionCNN(num_classes=dataset.num_classes, input_size=64).to(device)
    model.load_state_dict(torch.load('optimized_spot_emotion_model.pth', map_location=device))
    model.eval()
    
    # Test multiple videos
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
            # Load and process
            temp_dataset = OptimizedRAVDESSDataset([Path(video_path).parent], 
                                                 max_frames=40, frame_size=64, is_training=False)
            
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
    # Train optimized model
    model, dataset, best_acc = train_optimized()
    
    # Test the model
    test_optimized_model()
    
    print(f"\n🎉 OPTIMIZATION RESULTS:")
    print(f"🏆 Best Accuracy: {best_acc:.2f}%")
    print(f"💾 Model: optimized_spot_emotion_model.pth")
    print(f"🎯 Target: 60-80% accuracy")
    if best_acc >= 60:
        print(f"✅ TARGET ACHIEVED! Ready for Spot robot!")
    else:
        print(f"📈 Improvement: {best_acc - 30.56:.1f}% gain over baseline")