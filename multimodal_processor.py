"""
Multimodal Data Processor for RAVDESS
Extracts features from video and audio for Spot robot emotion recognition
Optimized for real-time processing on robot hardware
"""

import cv2
import numpy as np
import librosa
import torch
import torch.nn.functional as F
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class VideoProcessor:
    """Process video files to extract facial features"""
    
    def __init__(self, target_size=(48, 48), fps_target=10):
        self.target_size = target_size
        self.fps_target = fps_target
        
        # Initialize face detection (for real-time processing on Spot)
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except:
            print("⚠️  Face detection not available, using full frames")
            self.face_cascade = None
    
    def extract_frames(self, video_path, max_frames=30):
        """Extract frames from video file"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame sampling to get target number of frames
        if total_frames > max_frames:
            step = total_frames // max_frames
        else:
            step = 1
        
        frame_count = 0
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % step == 0:
                # Process frame
                processed_frame = self.process_frame(frame)
                if processed_frame is not None:
                    frames.append(processed_frame)
            
            frame_count += 1
        
        cap.release()
        
        # Convert to tensor [T, C, H, W] for temporal processing
        if frames:
            frames_tensor = torch.stack([torch.from_numpy(f).float() for f in frames])
            frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
            return frames_tensor
        else:
            return torch.zeros((1, 1, *self.target_size))
    
    def process_frame(self, frame):
        """Process individual frame - detect face, resize, normalize"""
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        if self.face_cascade is not None:
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Use the largest face
                face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = face
                
                # Add some padding
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(gray.shape[1] - x, w + 2 * padding)
                h = min(gray.shape[0] - y, h + 2 * padding)
                
                face_roi = gray[y:y+h, x:x+w]
            else:
                # Use center crop if no face detected
                h, w = gray.shape
                crop_size = min(h, w)
                start_h = (h - crop_size) // 2
                start_w = (w - crop_size) // 2
                face_roi = gray[start_h:start_h+crop_size, start_w:start_w+crop_size]
        else:
            # Use full frame if face detection not available
            face_roi = gray
        
        # Resize to target size
        resized = cv2.resize(face_roi, self.target_size)
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # Add channel dimension for grayscale
        normalized = np.expand_dims(normalized, axis=2)
        
        return normalized

class AudioProcessor:
    """Process audio files to extract features for emotion recognition"""
    
    def __init__(self, sample_rate=22050, n_mfcc=39, n_mels=96):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.hop_length = 512
        self.n_fft = 2048
    
    def load_audio(self, audio_path, duration=None):
        """Load and preprocess audio file"""
        try:
            # Load audio
            audio, sr = librosa.load(str(audio_path), sr=self.sample_rate, duration=duration)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            return audio, sr
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return np.zeros(self.sample_rate), self.sample_rate
    
    def extract_mfcc(self, audio):
        """Extract MFCC features"""
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        
        # Add delta and delta-delta features
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Combine features
        features = np.vstack([mfcc, delta, delta2])  # Shape: (117, time)
        
        return features.T  # Return as (time, features)
    
    def extract_logmel(self, audio):
        """Extract log-mel spectrogram features"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        
        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel.T  # Return as (time, n_mels)
    
    def extract_prosodic_features(self, audio):
        """Extract prosodic features (pitch, energy, etc.)"""
        features = {}
        
        # Fundamental frequency (pitch)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            hop_length=self.hop_length
        )
        
        # Replace NaN values
        f0 = np.nan_to_num(f0)
        
        # Energy/RMS
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)[0]
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate, hop_length=self.hop_length)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate, hop_length=self.hop_length)[0]
        
        # Combine all features
        prosodic = np.column_stack([
            f0[:len(rms)],  # Ensure same length
            rms,
            zcr[:len(rms)],
            spectral_centroids[:len(rms)],
            spectral_rolloff[:len(rms)]
        ])
        
        return prosodic
    
    def process_audio(self, audio_path, target_length=None):
        """Complete audio processing pipeline"""
        # Load audio
        audio, sr = self.load_audio(audio_path)
        
        # Extract features
        mfcc_features = self.extract_mfcc(audio)
        logmel_features = self.extract_logmel(audio)
        prosodic_features = self.extract_prosodic_features(audio)
        
        # Ensure same temporal dimension
        min_length = min(len(mfcc_features), len(logmel_features), len(prosodic_features))
        mfcc_features = mfcc_features[:min_length]
        logmel_features = logmel_features[:min_length]
        prosodic_features = prosodic_features[:min_length]
        
        # Target length processing (for sequence alignment)
        if target_length is not None:
            mfcc_features = self._adjust_length(mfcc_features, target_length)
            logmel_features = self._adjust_length(logmel_features, target_length)
            prosodic_features = self._adjust_length(prosodic_features, target_length)
        
        return {
            'mfcc': torch.from_numpy(mfcc_features).float(),
            'logmel': torch.from_numpy(logmel_features).float(),
            'prosodic': torch.from_numpy(prosodic_features).float()
        }
    
    def _adjust_length(self, features, target_length):
        """Adjust feature length to target by padding or truncating"""
        current_length = len(features)
        
        if current_length == target_length:
            return features
        elif current_length > target_length:
            # Truncate
            return features[:target_length]
        else:
            # Pad with zeros
            padding = np.zeros((target_length - current_length, features.shape[1]))
            return np.vstack([features, padding])

class MultimodalProcessor:
    """Combine video and audio processing for multimodal emotion recognition"""
    
    def __init__(self, target_sequence_length=100):
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()
        self.target_sequence_length = target_sequence_length
    
    def process_pair(self, video_path, audio_path):
        """Process a video-audio pair"""
        # Process video
        video_features = self.video_processor.extract_frames(video_path)
        
        # Process audio
        audio_features = self.audio_processor.process_audio(
            audio_path, 
            target_length=self.target_sequence_length
        )
        
        # Align temporal dimensions
        video_features = self._align_video_to_audio(video_features, self.target_sequence_length)
        
        return {
            'video': video_features,
            'audio': audio_features,
            'sequence_length': self.target_sequence_length
        }
    
    def _align_video_to_audio(self, video_tensor, target_length):
        """Align video sequence to audio sequence length"""
        current_length = video_tensor.shape[0]
        
        if current_length == target_length:
            return video_tensor
        elif current_length > target_length:
            # Uniformly sample frames
            indices = np.linspace(0, current_length - 1, target_length, dtype=int)
            return video_tensor[indices]
        else:
            # Interpolate to target length
            video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension
            aligned = F.interpolate(
                video_tensor.permute(0, 2, 3, 4, 1),  # [B, T, C, H, W] -> [B, C, H, W, T]
                size=target_length,
                mode='linear',
                align_corners=False
            )
            return aligned.permute(0, 4, 1, 2, 3).squeeze(0)  # Back to [T, C, H, W]

def test_processors():
    """Test the video and audio processors"""
    print("🧪 Testing Multimodal Processors")
    print("=" * 50)
    
    processor = MultimodalProcessor()
    
    # Test with sample files (if they exist)
    sample_video = Path("c:/Users/Ikean/RJCMA/Video_Song_Actor_02/Actor_02/01-02-03-01-01-01-02.mp4")
    sample_audio = Path("c:/Users/Ikean/RJCMA/Audio_Song_Actors_01-24/Actor_02/03-02-03-01-01-01-02.wav")
    
    if sample_video.exists() and sample_audio.exists():
        print(f"📹 Processing video: {sample_video.name}")
        print(f"🎵 Processing audio: {sample_audio.name}")
        
        result = processor.process_pair(sample_video, sample_audio)
        
        print(f"✅ Video features shape: {result['video'].shape}")
        print(f"✅ MFCC features shape: {result['audio']['mfcc'].shape}")
        print(f"✅ LogMel features shape: {result['audio']['logmel'].shape}")
        print(f"✅ Prosodic features shape: {result['audio']['prosodic'].shape}")
        print(f"🔄 Sequence length: {result['sequence_length']}")
    else:
        print("⚠️  Sample files not found, skipping test")
    
    return True

if __name__ == "__main__":
    test_processors()