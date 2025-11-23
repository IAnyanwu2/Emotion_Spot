#!/usr/bin/env python3
"""
Inspect the saved model to understand its architecture
"""

import torch
import torch.nn as nn
from pathlib import Path

def inspect_saved_model():
    """Inspect the architecture of the saved model"""
    print("🔍 INSPECTING SAVED MODEL ARCHITECTURE")
    print("=" * 50)
    
    # Load the state dict
    model_path = 'spot_emotion_model_6class.pth'
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return
    
    state_dict = torch.load(model_path, map_location='cpu')
    
    print("📊 Model layers and shapes:")
    for name, tensor in state_dict.items():
        print(f"   {name}: {tensor.shape}")
    
    print(f"\n📈 Total parameters: {sum(p.numel() for p in state_dict.values()):,}")
    
    # Try to infer the architecture
    print("\n🤖 Inferring architecture...")
    
    # Check for common layer patterns
    has_features = any(name.startswith('features') for name in state_dict.keys())
    has_temporal_lstm = any('temporal_lstm' in name for name in state_dict.keys())
    has_classifier = any(name.startswith('classifier') for name in state_dict.keys())
    
    print(f"   Has 'features' module: {has_features}")
    print(f"   Has 'temporal_lstm': {has_temporal_lstm}")
    print(f"   Has 'classifier' module: {has_classifier}")
    
    # Look at specific layers
    if 'features.0.weight' in state_dict:
        conv1_shape = state_dict['features.0.weight'].shape
        print(f"   First conv layer: {conv1_shape}")
    
    if 'classifier.1.weight' in state_dict:
        fc_shape = state_dict['classifier.1.weight'].shape
        print(f"   First classifier layer: {fc_shape}")
    
    if 'temporal_lstm.weight_ih_l0' in state_dict:
        lstm_ih_shape = state_dict['temporal_lstm.weight_ih_l0'].shape
        print(f"   LSTM input-hidden weights: {lstm_ih_shape}")

class ActualModelArchitecture(nn.Module):
    """Recreate the actual architecture based on inspection"""
    def __init__(self, num_classes=6):
        super().__init__()
        
        # Based on the layer names, it looks like this structure:
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # features.0
            nn.BatchNorm2d(64),                          # features.1
            nn.ReLU(inplace=True),                       # features.2
            nn.MaxPool2d(2, 2),                          # features.3
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # features.4
            nn.BatchNorm2d(128),                         # features.5
            nn.ReLU(inplace=True),                       # features.6
            nn.MaxPool2d(2, 2),                          # features.7
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # features.8
            nn.BatchNorm2d(256),                         # features.9
            nn.ReLU(inplace=True),                       # features.10
            nn.MaxPool2d(2, 2),                          # features.11
            nn.Conv2d(256, 512, kernel_size=3, padding=1), # features.12
            nn.BatchNorm2d(512),                         # features.13
            nn.ReLU(inplace=True),                       # features.14
            nn.AdaptiveAvgPool2d((1, 1))                 # features.15
        )
        
        # Bidirectional LSTM based on reverse weights
        self.temporal_lstm = nn.LSTM(512, 256, batch_first=True, bidirectional=True)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),                             # classifier.0
            nn.Linear(512, 256),                         # classifier.1
            nn.ReLU(inplace=True),                       # classifier.2
            nn.Dropout(0.3),                             # classifier.3
            nn.Linear(256, num_classes)                  # classifier.4
        )
    
    def forward(self, x):
        b, t, h, w, c = x.shape
        x = x.view(b * t, c, h, w)
        
        # Feature extraction
        x = self.features(x)  # (b*t, 512, 1, 1)
        x = x.view(b * t, -1)  # (b*t, 512)
        
        # Reshape for temporal processing
        x = x.view(b, t, -1)  # (b, t, 512)
        
        # LSTM processing
        lstm_out, _ = self.temporal_lstm(x)  # (b, t, 512)
        
        # Use last output or mean
        x = lstm_out[:, -1, :]  # (b, 512)
        
        # Classification
        x = self.classifier(x)
        return x

def test_architecture_match():
    """Test if our recreated architecture matches the saved model"""
    print("\n🔄 TESTING ARCHITECTURE MATCH")
    print("=" * 40)
    
    # Create model and load weights
    model = ActualModelArchitecture(num_classes=6)
    
    try:
        state_dict = torch.load('spot_emotion_model_6class.pth', map_location='cpu')
        model.load_state_dict(state_dict)
        print("✅ Successfully loaded saved model!")
        
        # Test with dummy input
        dummy_input = torch.randn(1, 30, 48, 48, 3)  # batch, time, height, width, channels
        output = model(dummy_input)
        print(f"✅ Model forward pass successful: {output.shape}")
        print(f"📊 Output probabilities: {torch.softmax(output, dim=1)}")
        
        return model
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

if __name__ == "__main__":
    inspect_saved_model()
    model = test_architecture_match()
    
    if model is not None:
        print("\n🎉 SUCCESS! Architecture correctly identified!")
        print("✅ Ready to create improvement script with correct architecture")
    else:
        print("\n❌ Need to adjust architecture based on inspection results")