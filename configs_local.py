# Local configuration for FER2013 dataset
# Simplified version for local CPU/GPU execution

import os
import torch

# Emotion categories for FER2013
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = 7

# Model dimensions
VIDEO_EMBEDDING_DIM = 512
VIDEO_TEMPORAL_DIM = 128

config = {
    # Device configuration - automatically detects GPU/CPU
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    
    # Dataset paths - adjusted for your local structure
    "dataset_path": "c:/Users/Ikean/RJCMA/fer2013",
    "train_path": "c:/Users/Ikean/RJCMA/fer2013/train",
    "test_path": "c:/Users/Ikean/RJCMA/fer2013/test",
    
    # Data frequency settings
    "frequency": {
        "video": None,
        "continuous_label": None,
    },

    # Data multipliers
    "multiplier": {
        "video": 1,
        "continuous_label": 1,
    },

    # Feature dimensions for FER2013
    "feature_dimension": {
        "video": (48, 48, 1),  # FER2013 images are 48x48 grayscale
        "continuous_label": (7,),  # 7 emotion categories
    },

    # Dataset information
    "dataset_info": {
        "name": "FER2013",
        "num_classes": NUM_CLASSES,
        "emotions": EMOTION_LABELS,
        "image_size": (48, 48),
        "channels": 1,  # Grayscale
    },

    # Training parameters for local execution
    "training": {
        "batch_size": 8,  # Smaller batch size for local training
        "learning_rate": 1e-4,
        "num_epochs": 50,  # Reduced for testing
        "early_stopping": 10,
        "window_length": 150,  # Reduced for local execution
        "hop_length": 100,
    },

    # Model parameters
    "model": {
        "num_heads": 2,
        "modal_dim": 32,
        "tcn_kernel_size": 3,  # Smaller kernel for efficiency
    }
}

# Function to get configuration
def get_local_config():
    """Get configuration optimized for local execution"""
    device_name = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"🖥️  Running on: {device_name}")
    print(f"📊 Dataset: {config['dataset_info']['name']}")
    print(f"📁 Dataset path: {config['dataset_path']}")
    return config