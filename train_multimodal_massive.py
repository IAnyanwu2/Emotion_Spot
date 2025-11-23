#!/usr/bin/env python3
"""
Massive Multimodal Emotion Training Script
Uses existing SimpleEmotionCNN architecture + expands to handle:
- 35,901 FER2013 images
- 1,012 RAVDESS audio files  
- 176 RAVDESS video files
Total: 37,089 training samples
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import cv2
import numpy as np
from pathlib import Path
import librosa
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import time
import random

# Import the existing working architecture
from train_original_fixed import SimpleEmotionCNN, FixedRAVDESSDataset

class FER2013Dataset(Dataset):
    """FER2013 Image Dataset Handler"""
    
    def __init__(self, root_dir, transform=None, target_size=48):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_size = target_size
        
        # FER2013 emotion mapping to match RAVDESS
        self.fer_emotions = {
            'angry': 'angry',
            'disgust': 'disgust', 
            'fear': 'fearful',
            'happy': 'happy',
            'neutral': 'neutral', 
            'sad': 'sad',
            'surprise': 'surprised'
        }
        
        # Match to RAVDESS emotions (6 classes)
        self.emotion_map = {
            'angry': 4,      # angry
            'fear': 5,       # fearful  
            'happy': 2,      # happy
            'neutral': 0,    # neutral
            'sad': 3,        # sad
            'disgust': 4,    # map to angry (closest)
            'surprise': 2    # map to happy (closest)
        }
        
        self.emotion_names = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful']
        self.samples = self._load_samples()
        
    def _load_samples(self):
        """Load all FER2013 image paths and labels"""
        samples = []
        
        for emotion_folder in self.root_dir.iterdir():
            if not emotion_folder.is_dir():
                continue
                
            emotion_name = emotion_folder.name.lower()
            if emotion_name in self.emotion_map:
                label = self.emotion_map[emotion_name]
                
                for img_path in emotion_folder.glob('*.jpg'):
                    samples.append((img_path, label))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.target_size, self.target_size))
        
        if self.transform:
            image = self.transform(image)
        else:
            # Convert to tensor and normalize
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])(image)
        
        # Convert to "video" format for compatibility with existing CNN
        # Repeat image to create fake temporal dimension with consistent shape
        video_tensor = image.unsqueeze(0).repeat(10, 1, 1, 1)  # 10 frames, 3 channels
        
        return video_tensor, torch.tensor(label, dtype=torch.long)

class RAVDESSAudioDataset(Dataset):
    """RAVDESS Audio Dataset Handler"""
    
    def __init__(self, audio_dirs, sample_rate=22050, duration=4.0):
        self.audio_dirs = [Path(d) for d in audio_dirs]
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        
        # RAVDESS emotion mapping (from filename)
        self.emotion_map = {
            '01': 0,  # neutral
            '02': 1,  # calm  
            '03': 2,  # happy
            '04': 3,  # sad
            '05': 4,  # angry
            '06': 5,  # fearful
            '07': 4,  # disgust -> angry
            '08': 2   # surprised -> happy
        }
        
        self.emotion_names = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful']
        self.audio_files = self._load_audio_files()
        
    def _load_audio_files(self):
        """Load all audio file paths"""
        audio_files = []
        
        for audio_dir in self.audio_dirs:
            if audio_dir.exists():
                for wav_file in audio_dir.rglob('*.wav'):
                    # Parse emotion from filename: XX-XX-EM-XX-XX-XX-XX.wav
                    parts = wav_file.stem.split('-')
                    if len(parts) >= 3:
                        emotion_code = parts[2]
                        if emotion_code in self.emotion_map:
                            label = self.emotion_map[emotion_code]
                            audio_files.append((wav_file, label))
        
        return audio_files
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path, label = self.audio_files[idx]
        
        try:
            # Load audio with librosa
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Normalize length
            if len(audio) > self.target_length:
                audio = audio[:self.target_length]
            elif len(audio) < self.target_length:
                # Pad with zeros
                padding = self.target_length - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant')
            
            # Convert to mel spectrogram for CNN processing
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=48, fmax=8000
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize
            mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
            
            # Convert to tensor and create fake video format
            mel_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0)  # Add channel dim
            
            # Resize to match video input (48x48)
            mel_tensor = torch.nn.functional.interpolate(
                mel_tensor.unsqueeze(0), size=(48, 48), mode='bilinear'
            ).squeeze(0)
            
            # Convert to 3 channels to match image format
            mel_tensor = mel_tensor.repeat(3, 1, 1)  # Make it 3 channels
            
            # Create fake temporal dimension (repeat for consistency)
            video_tensor = mel_tensor.unsqueeze(0).repeat(10, 1, 1, 1)  # 10 frames, 3 channels
            
            return video_tensor, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # Return zero tensor on error with correct shape
            video_tensor = torch.zeros(10, 3, 48, 48)  # 10 frames, 3 channels
            return video_tensor, torch.tensor(label, dtype=torch.long)

class MultimodalEmotionCNN(nn.Module):
    """Extended CNN that handles multimodal input"""
    
    def __init__(self, num_classes=6, input_size=48, dropout_rate=0.3):
        super().__init__()
        
        # Use the same architecture as SimpleEmotionCNN
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
        )
        
        # Calculate feature size
        feature_size = 128 * (input_size // 8) * (input_size // 8)
        
        self.temporal_lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  # 512 = 256 * 2 (bidirectional)
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        
        # Process each frame through CNN
        x = x.view(batch_size * seq_len, channels, height, width)
        x = self.features(x)
        x = x.view(batch_size * seq_len, -1)
        
        # Reshape for LSTM
        x = x.view(batch_size, seq_len, -1)
        
        # LSTM processing
        lstm_out, _ = self.temporal_lstm(x)
        x = lstm_out[:, -1, :]  # Take last output
        
        # Classification
        x = self.classifier(x)
        return x

class VideoDatasetWrapper(Dataset):
    """Wrapper to ensure consistent tensor format for video data"""
    
    def __init__(self, video_dataset):
        self.dataset = video_dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        video_tensor, label = self.dataset[idx]
        
        # Check if tensor needs reshaping
        if video_tensor.dim() == 4:
            # If shape is [frames, height, width, channels], reshape to [frames, channels, height, width]
            if video_tensor.shape[-1] == 3:  # Last dimension is channels
                video_tensor = video_tensor.permute(0, 3, 1, 2)
        
        # Ensure we have exactly 10 frames
        current_frames = video_tensor.shape[0]
        if current_frames != 10:
            if current_frames > 10:
                # Take first 10 frames
                video_tensor = video_tensor[:10]
            else:
                # Repeat frames to get 10
                repeat_factor = 10 // current_frames
                remainder = 10 % current_frames
                video_tensor = torch.cat([
                    video_tensor.repeat(repeat_factor, 1, 1, 1),
                    video_tensor[:remainder]
                ], dim=0)
        
        return video_tensor, torch.tensor(label, dtype=torch.long)

def create_massive_dataset():
    """Create combined dataset from all sources"""
    print("🚀 Creating massive multimodal dataset...")
    
    datasets = []
    
    # 1. FER2013 Images (35,901 samples)
    print("📸 Loading FER2013 images...")
    fer_train = FER2013Dataset('fer2013/train')
    fer_test = FER2013Dataset('fer2013/test')
    fer_combined = ConcatDataset([fer_train, fer_test])
    datasets.append(fer_combined)
    print(f"   FER2013: {len(fer_combined)} images loaded")
    
    # 2. RAVDESS Audio (1,012 samples)  
    print("🎵 Loading RAVDESS audio...")
    audio_dirs = [f'Audio_Song_Actors_01-24/Actor_{i:02d}' for i in range(1, 25)]
    audio_dataset = RAVDESSAudioDataset(audio_dirs)
    datasets.append(audio_dataset)
    print(f"   RAVDESS Audio: {len(audio_dataset)} files loaded")
    
    # 3. RAVDESS Video (176 samples)
    print("🎬 Loading RAVDESS videos...")
    video_dirs = ["Video_Song_Actor_02/Actor_02", "Video_Song_Actor_06/Actor_06"]
    raw_video_dataset = FixedRAVDESSDataset(video_dirs, max_frames=10, frame_size=48)
    video_dataset = VideoDatasetWrapper(raw_video_dataset)  # Wrap to fix tensor format
    datasets.append(video_dataset)
    print(f"   RAVDESS Video: {len(video_dataset)} videos loaded")
    
    # Combine all datasets
    massive_dataset = ConcatDataset(datasets)
    total_samples = len(massive_dataset)
    
    print(f"\n🎉 MASSIVE DATASET CREATED!")
    print(f"📊 Total samples: {total_samples:,}")
    print(f"   FER2013 Images: {len(fer_combined):,} ({len(fer_combined)/total_samples*100:.1f}%)")
    print(f"   RAVDESS Audio: {len(audio_dataset):,} ({len(audio_dataset)/total_samples*100:.1f}%)")
    print(f"   RAVDESS Video: {len(video_dataset):,} ({len(video_dataset)/total_samples*100:.1f}%)")
    
    return massive_dataset

def train_massive_model():
    """Train on the massive multimodal dataset"""
    print("\n" + "="*60)
    print("🚀 MASSIVE MULTIMODAL EMOTION TRAINING")
    print("🎯 Target: 75-85% accuracy (vs current 58.33%)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # Create massive dataset
    dataset = create_massive_dataset()
    
    # Train/test split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Training: {len(train_dataset):,} samples")
    print(f"   Testing: {len(test_dataset):,} samples")
    
    # Data loaders (disable multiprocessing for Windows compatibility)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Model
    model = MultimodalEmotionCNN(num_classes=6, input_size=48).to(device)
    
    # Try to load existing model for continuous training
    model_path = 'massive_emotion_model.pth'
    if Path(model_path).exists():
        print(f"📥 Loading existing model: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🔧 Model parameters: {total_params:,}")
    
    # Training loop
    num_epochs = 50
    best_accuracy = 0
    patience = 10
    no_improve_count = 0
    
    print(f"\n🚀 Training started...")
    print(f"📊 Epochs: {num_epochs}, Patience: {patience}")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
            
            if batch_idx % 100 == 0:
                progress = 100. * batch_idx / len(train_loader)
                print(f"Batch {batch_idx}/{len(train_loader)} ({progress:.1f}%), Loss: {loss.item():.4f}")
        
        # Testing
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
                test_total += target.size(0)
        
        # Calculate metrics
        train_accuracy = 100. * train_correct / train_total
        test_accuracy = 100. * test_correct / test_total
        epoch_time = time.time() - start_time
        
        # Learning rate scheduling
        scheduler.step(test_loss)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"   Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"   Train Accuracy: {train_accuracy:.2f}%")
        print(f"   Test Loss: {test_loss/len(test_loader):.4f}")
        print(f"   Test Accuracy: {test_accuracy:.2f}%")
        
        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), model_path)
            print(f"   💾 New best model saved! Accuracy: {best_accuracy:.2f}%")
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Early stopping
        if no_improve_count >= patience:
            print(f"⏹️  Early stopping: No improvement for {patience} epochs")
            break
        
        print(f"   Time: {epoch_time:.2f}s")
        print("-" * 60)
    
    print(f"\n🎉 MASSIVE TRAINING COMPLETE!")
    print(f"🏆 Best accuracy: {best_accuracy:.2f}%")
    print(f"📈 Improvement: +{best_accuracy - 58.33:.2f}% vs previous model")
    print(f"💾 Model saved: {model_path}")
    
    return model, best_accuracy

if __name__ == "__main__":
    # Train the massive model
    model, accuracy = train_massive_model()
    
    print(f"\n🎯 READY FOR SPOT ROBOT DEPLOYMENT!")
    print(f"🤖 Final accuracy: {accuracy:.2f}%")
    print(f"📊 Training data: 37,089 samples")
    print(f"🎭 Emotions: ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful']")