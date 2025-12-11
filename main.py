"""
Main Program - Beginner Version
This is the main program to detect deepfake audio
"""

import os
import torch
import numpy as np
import librosa

# Import our model components
from audio_encoder import AudioEncoder
from anomaly_extractor import AnomalyFeatureExtractor
from classifier import SimpleClassifier, DeepfakeDetector



# Audio Preprocessing Functions


def load_audio_file(file_path, target_sample_rate=16000):
    """
    Load an audio file and convert it to 16kHz mono
    
    Returns: audio array, sample rate
    """
    audio, sample_rate = librosa.load(file_path, sr=target_sample_rate, mono=True)
    return audio, sample_rate


def normalize_audio(audio):
    """
    Normalize audio to range [-1, 1]
    """
    max_value = np.abs(audio).max()
    if max_value > 0:
        audio = audio / max_value
    return audio


def trim_silence(audio, sample_rate, top_db=20):
    """
    Remove silence from beginning and end of audio
    """
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed_audio


def create_mel_spectrogram(audio, sample_rate, n_mels=80):
    """
    Convert audio to mel-spectrogram
    
    Mel-spectrogram is like a picture of the sound showing:
    - Time on x-axis
    - Frequency on y-axis
    - Intensity as brightness
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=512,
        hop_length=160,
        win_length=400
    )
    
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Transpose to (time, frequency)
    mel_spec_db = mel_spec_db.T
    
    return mel_spec_db



# Main Detector Class


class SimpleDeepfakeDetector:
    """
    This is the main class for detecting deepfakes.
    
    Usage:
        detector = SimpleDeepfakeDetector()
        result = detector.check_audio("my_audio.wav")
        print(result['label'])  # "REAL" or "FAKE"
    """
    
    def __init__(self, model_weights_path=None):
        """
        Initialize the detector
        
        Args:
            model_weights_path: path to saved model weights (optional)
        """
        # Check if GPU is available
        if torch.cuda.is_available():
            self.device = 'cuda'
            print("Using GPU")
        else:
            self.device = 'cpu'
            print("Using CPU")
        
        self.sample_rate = 16000
        
        print("\nBuilding model...")
        
        # Create audio encoder
        self.audio_encoder = AudioEncoder(
            mel_bands=80,
            cnn_size=256,
            num_cnn_layers=2,
            embedding_size=256,
            num_heads=8,
            num_transformer_layers=4,
            feedforward_size=1024,
            dropout_rate=0.1
        )
        
        # Create anomaly feature extractor
        self.anomaly_extractor = AnomalyFeatureExtractor(
            sample_rate=self.sample_rate,
            output_size=40
        )
        
        # Create complete model
        self.model = DeepfakeDetector(
            audio_encoder=self.audio_encoder,
            anomaly_extractor=self.anomaly_extractor,
            audio_feature_size=256,
            anomaly_feature_size=40,
            hidden_size=256,
            dropout_rate=0.3
        )
        
        # Move model to device (GPU or CPU)
        self.model.to(self.device)
        
        # Load weights if provided
        if model_weights_path and os.path.exists(model_weights_path):
            self.load_weights(model_weights_path)
            print(f"Loaded weights from: {model_weights_path}")
        else:
            print("Using random weights (not trained yet)")
        
        # Set model to evaluation mode
        self.model.eval()
        
        print("Model ready!\n")
    
    def load_weights(self, path):
        """Load model weights from file"""
        checkpoint = torch.load(path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
    
    def save_weights(self, path):
        """Save model weights to file"""
        torch.save({
            'model_state_dict': self.model.state_dict()
        }, path)
        print(f"Weights saved to: {path}")
    
    def prepare_audio(self, audio_file):
        """
        Load and prepare audio for the model
        
        Returns: (mel_spectrogram, raw_audio)
        """
        print(f"Loading audio: {audio_file}")
        
        # Load audio
        audio, sr = load_audio_file(audio_file, self.sample_rate)
        
        # Normalize
        audio = normalize_audio(audio)
        
        # Trim silence
        audio = trim_silence(audio, sr)
        
        # Make sure audio is long enough (at least 0.5 seconds)
        min_length = int(self.sample_rate * 0.5)
        if len(audio) < min_length:
            # Pad with zeros if too short
            audio = np.pad(audio, (0, min_length - len(audio)))
        
        # Create mel-spectrogram
        mel_spec = create_mel_spectrogram(audio, sr)
        
        # Convert to tensors
        mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)  # Add batch dimension
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0)   # Add batch dimension
        
        return mel_tensor, audio_tensor
    
    def check_audio(self, audio_file, threshold=0.5):
        """
        Check if an audio file is a deepfake
        
        Args:
            audio_file: path to audio file
            threshold: decision threshold (default: 0.5)
        
        Returns:
            Dictionary with results
        """
        # Check if file exists
        if not os.path.exists(audio_file):
            return {'error': f"File not found: {audio_file}"}
        
        # Prepare audio
        mel_spec, raw_audio = self.prepare_audio(audio_file)
        
        # Move to device
        mel_spec = mel_spec.to(self.device)
        raw_audio = raw_audio.to(self.device)
        
        # Make prediction (no gradient needed)
        with torch.no_grad():
            fake_score = self.model(mel_spec, raw_audio)
        
        # Get score as a number
        score = fake_score.item()
        
        # Decide label
        if score > threshold:
            label = "FAKE"
        else:
            label = "REAL"
        
        # Calculate confidence
        # If score is far from 0.5, we're more confident
        confidence = abs(score - 0.5) * 2  # Scale to 0-1
        
        # Return results
        return {
            'file': audio_file,
            'label': label,
            'fake_score': score,
            'confidence': confidence
        }
    
    def check_multiple_files(self, audio_files, threshold=0.5):
        """
        Check multiple audio files
        
        Returns: list of results
        """
        results = []
        
        for audio_file in audio_files:
            try:
                result = self.check_audio(audio_file, threshold)
                results.append(result)
            except Exception as e:
                results.append({
                    'file': audio_file,
                    'error': str(e)
                })
        
        return results



# Simple Function for Easy Use


# Global detector (created once, reused)
global_detector = None

def detect_deepfake(audio_file, model_path=None, threshold=0.5):
    """
    Simple function to detect deepfakes
    
    Example:
        result = detect_deepfake("my_audio.wav")
        print(result['label'])
        print(result['fake_score'])
    """
    global global_detector
    
    # Create detector if not already created
    if global_detector is None:
        global_detector = SimpleDeepfakeDetector(model_path)
    
    # Check the audio
    return global_detector.check_audio(audio_file, threshold)



# Command Line Interface


def main():
    """
    Run from command line
    
    Usage:
        python main.py --audio test.wav
        python main.py --audio audio_folder/ --batch
    """
    import argparse
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(
        description='Simple Deepfake Audio Detector'
    )
    
    parser.add_argument(
        '--audio',
        type=str,
        required=True,
        help='Path to audio file or folder'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to model weights'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Decision threshold (default: 0.5)'
    )
    
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all files in folder'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 60)
    print("DEEPFAKE AUDIO DETECTOR")
    print("=" * 60)
    
    # Create detector
    detector = SimpleDeepfakeDetector(args.model)
    
    if args.batch:
        # Process multiple files
        if not os.path.isdir(args.audio):
            print(f"Error: {args.audio} is not a folder")
            return
        
        # Find all audio files
        audio_files = []
        for file in os.listdir(args.audio):
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                audio_files.append(os.path.join(args.audio, file))
        
        print(f"Found {len(audio_files)} audio files\n")
        
        # Check each file
        for audio_file in audio_files:
            result = detector.check_audio(audio_file, args.threshold)
            
            if 'error' in result:
                print(f"❌ {os.path.basename(audio_file)}: {result['error']}")
            else:
                icon = "🔴" if result['label'] == "FAKE" else "🟢"
                print(f"{icon} {os.path.basename(audio_file)}")
                print(f"   Label: {result['label']}")
                print(f"   Score: {result['fake_score']:.4f}")
                print(f"   Confidence: {result['confidence']:.2%}\n")
    
    else:
        # Process single file
        result = detector.check_audio(args.audio, args.threshold)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print("\nRESULTS:")
            print("=" * 40)
            print(f"File: {result['file']}")
            print(f"Label: {result['label']}")
            print(f"Fake Score: {result['fake_score']:.4f}")
            print(f"Confidence: {result['confidence']:.2%}")
            print("=" * 40)
            
            if result['label'] == "FAKE":
                print("\n⚠️  WARNING: This audio is likely FAKE/SYNTHETIC")
            else:
                print("\n✅ This audio appears to be REAL/AUTHENTIC")
    
    print("\nDone!")



# RUN THE PROGRAM


if __name__ == "__main__":
    main()