"""
RAVDESS Dataset Handler for Spot Robot Emotion Recognition
Handles multimodal (video + audio) emotion data for real-time processing

RAVDESS Filename Structure: Modality-Vocal_channel-Emotion-Emotional_intensity-Statement-Repetition-Actor
Example: 01-02-03-01-01-02-02.mp4 = Video, Speech, Happy, Normal, Statement1, Rep2, Actor02
"""

import os
import re
import pandas as pd
from pathlib import Path
import torch
import numpy as np

class RAVDESSDataset:
    def __init__(self, data_root="c:/Users/Ikean/RJCMA"):
        self.data_root = Path(data_root)
        self.video_paths = []
        self.audio_paths = []
        
        # RAVDESS emotion mapping for Spot robot
        self.emotion_map = {
            1: "neutral",    # Calm baseline for robot
            2: "calm",       # Relaxed state
            3: "happy",      # Positive interaction
            4: "sad",        # Sympathy/comfort mode
            5: "angry",      # Alert/defensive mode
            6: "fearful",    # Caution mode
            7: "disgust",    # Avoidance behavior
            8: "surprised"   # Attention/investigation mode
        }
        
        # Modality mapping
        self.modality_map = {
            1: "video-only",
            2: "audio-only", 
            3: "audio-video"
        }
        
        # Initialize dataset
        self._scan_dataset()
    
    def _scan_dataset(self):
        """Scan for RAVDESS video and audio files"""
        print("🔍 Scanning RAVDESS dataset...")
        
        # Scan video directories
        video_dirs = [
            self.data_root / "Video_Song_Actor_02",
            self.data_root / "Video_Song_Actor_06"
        ]
        
        for video_dir in video_dirs:
            if video_dir.exists():
                for actor_dir in video_dir.iterdir():
                    if actor_dir.is_dir():
                        for video_file in actor_dir.glob("*.mp4"):
                            self.video_paths.append(video_file)
        
        # Scan audio directory
        audio_dir = self.data_root / "Audio_Song_Actors_01-24"
        if audio_dir.exists():
            for actor_dir in audio_dir.iterdir():
                if actor_dir.is_dir():
                    for audio_file in actor_dir.glob("*.wav"):
                        self.audio_paths.append(audio_file)
        
        print(f"✅ Found {len(self.video_paths)} video files")
        print(f"✅ Found {len(self.audio_paths)} audio files")
    
    def parse_filename(self, filename):
        """Parse RAVDESS filename to extract metadata"""
        # Remove extension and split by dashes
        name_parts = Path(filename).stem.split('-')
        
        if len(name_parts) != 7:
            return None
        
        try:
            metadata = {
                'modality': int(name_parts[0]),
                'vocal_channel': int(name_parts[1]),
                'emotion': int(name_parts[2]),
                'intensity': int(name_parts[3]),
                'statement': int(name_parts[4]),
                'repetition': int(name_parts[5]),
                'actor': int(name_parts[6]),
                'emotion_label': self.emotion_map.get(int(name_parts[2]), 'unknown'),
                'filename': filename
            }
            return metadata
        except ValueError:
            return None
    
    def create_dataset_info(self):
        """Create dataset information for the RCMA model"""
        dataset_info = {
            'video_files': [],
            'audio_files': [],
            'emotions': list(self.emotion_map.values()),
            'num_emotions': len(self.emotion_map),
            'actors': set(),
            'paired_files': []  # Video-audio pairs
        }
        
        # Process video files
        for video_path in self.video_paths:
            metadata = self.parse_filename(video_path.name)
            if metadata:
                dataset_info['video_files'].append({
                    'path': str(video_path),
                    'metadata': metadata
                })
                dataset_info['actors'].add(metadata['actor'])
        
        # Process audio files  
        for audio_path in self.audio_paths:
            metadata = self.parse_filename(audio_path.name)
            if metadata:
                dataset_info['audio_files'].append({
                    'path': str(audio_path),
                    'metadata': metadata
                })
                dataset_info['actors'].add(metadata['actor'])
        
        # Create video-audio pairs for multimodal training
        dataset_info['paired_files'] = self._create_pairs(dataset_info)
        dataset_info['actors'] = sorted(list(dataset_info['actors']))
        
        return dataset_info
    
    def _create_pairs(self, dataset_info):
        """Create video-audio pairs based on matching metadata"""
        pairs = []
        
        # Create lookup for audio files
        audio_lookup = {}
        for audio_entry in dataset_info['audio_files']:
            meta = audio_entry['metadata']
            # Create key for matching (emotion, actor, statement, repetition)
            key = (meta['emotion'], meta['actor'], meta['statement'], meta['repetition'])
            audio_lookup[key] = audio_entry
        
        # Match video files with corresponding audio
        for video_entry in dataset_info['video_files']:
            meta = video_entry['metadata']
            key = (meta['emotion'], meta['actor'], meta['statement'], meta['repetition'])
            
            if key in audio_lookup:
                pairs.append({
                    'video': video_entry,
                    'audio': audio_lookup[key],
                    'emotion': meta['emotion'],
                    'emotion_label': meta['emotion_label'],
                    'actor': meta['actor'],
                    'intensity': meta['intensity']
                })
        
        return pairs
    
    def get_spot_emotion_mapping(self):
        """Get emotion mapping optimized for Spot robot behaviors"""
        return {
            'neutral': {'behavior': 'idle', 'confidence_threshold': 0.7},
            'calm': {'behavior': 'relaxed_stance', 'confidence_threshold': 0.6},
            'happy': {'behavior': 'playful_bounce', 'confidence_threshold': 0.8},
            'sad': {'behavior': 'comfort_approach', 'confidence_threshold': 0.7},
            'angry': {'behavior': 'alert_stance', 'confidence_threshold': 0.9},
            'fearful': {'behavior': 'cautious_step_back', 'confidence_threshold': 0.8},
            'disgust': {'behavior': 'head_turn_away', 'confidence_threshold': 0.7},
            'surprised': {'behavior': 'head_tilt_investigate', 'confidence_threshold': 0.6}
        }
    
    def export_for_training(self, output_path="ravdess_dataset_info.pkl"):
        """Export dataset info for RCMA training"""
        import pickle
        
        dataset_info = self.create_dataset_info()
        
        # Add configuration for RCMA
        dataset_info['config'] = {
            'video_size': (48, 48),  # Resize to match FER2013 format
            'audio_features': ['mfcc', 'logmel'],
            'sample_rate': 22050,
            'n_mfcc': 39,
            'hop_length': 512,
            'window_length': 300,  # For sequence processing
            'emotions_count': 8,    # 8 RAVDESS emotions
            'modalities': ['video', 'audio'],
            'spot_integration': True
        }
        
        # Save dataset info
        output_file = self.data_root / output_path
        with open(output_file, 'wb') as f:
            pickle.dump(dataset_info, f)
        
        print(f"📁 Dataset info exported to: {output_file}")
        print(f"📊 Total multimodal pairs: {len(dataset_info['paired_files'])}")
        print(f"🎭 Emotions: {dataset_info['emotions']}")
        print(f"👥 Actors: {len(dataset_info['actors'])}")
        
        return dataset_info

def test_ravdess_dataset():
    """Test the RAVDESS dataset loading"""
    print("🧪 Testing RAVDESS Dataset Handler")
    print("=" * 50)
    
    dataset = RAVDESSDataset()
    dataset_info = dataset.export_for_training()
    
    # Display sample data
    if dataset_info['paired_files']:
        sample = dataset_info['paired_files'][0]
        print("\n📝 Sample data pair:")
        print(f"   Video: {Path(sample['video']['path']).name}")
        print(f"   Audio: {Path(sample['audio']['path']).name}")
        print(f"   Emotion: {sample['emotion_label']}")
        print(f"   Actor: {sample['actor']}")
    
    # Show Spot robot emotion mapping
    spot_mapping = dataset.get_spot_emotion_mapping()
    print("\n🤖 Spot Robot Emotion Behaviors:")
    for emotion, config in spot_mapping.items():
        print(f"   {emotion} → {config['behavior']} (confidence: {config['confidence_threshold']})")
    
    return dataset_info

if __name__ == "__main__":
    test_ravdess_dataset()