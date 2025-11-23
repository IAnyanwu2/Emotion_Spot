# Configuration for FER2013 dataset
# This replaces the ABAW6 configuration

VIDEO_EMBEDDING_DIM = 512
BERT_DIM = 768  # If you still want to use text features
VIDEO_TEMPORAL_DIM = 128
BERT_TEMPORAL_DIM = 512

config = {
    "frequency": {
        "video": None,
        "continuous_label": None,
        "bert": None  # Optional: for text-based emotion analysis
    },

    "multiplier": {
        "video": 1,
        "continuous_label": 1,
        "bert": 1,
    },

    "feature_dimension": {
        "video": (48, 48, 1),  # FER2013 images are 48x48 grayscale
        "continuous_label": (7,),  # 7 emotion categories
        "bert": (768,)  # If using text features
    },

    "dataset_info": {
        "name": "FER2013",
        "num_classes": 7,
        "class_names": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"],
        "class_mapping": {
            "angry": 0,
            "disgust": 1, 
            "fear": 2,
            "happy": 3,
            "sad": 4,
            "surprise": 5,
            "neutral": 6
        }
    },

    # FER2013 specific paths
    "root_directory": "datasets/FER2013",
    "train_folder": "train",
    "test_folder": "test",
    
    # Image processing
    "image_size": 48,
    "crop_size": 48,
    "mean": [0.5],  # Grayscale mean
    "std": [0.5],   # Grayscale std
    
    # Since FER2013 doesn't have valence-arousal, map emotions to VA space
    "emotion_to_va": {
        "angry": [-0.6, 0.6],      # Negative valence, high arousal
        "disgust": [-0.7, 0.4],    # Very negative valence, medium arousal  
        "fear": [-0.5, 0.8],       # Negative valence, very high arousal
        "happy": [0.8, 0.6],       # Very positive valence, high arousal
        "sad": [-0.6, -0.4],       # Negative valence, low arousal
        "surprise": [0.2, 0.8],    # Slightly positive, very high arousal
        "neutral": [0.0, 0.0]      # Neutral valence and arousal
    }
}

# VA estimation labels (if you want to train for valence-arousal)
VA_LABELS = ["valence", "arousal"]