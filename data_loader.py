"""
Dataset Handler for Deepfake Audio Detection
Loads real and fake audio files for training
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
from pathlib import Path


class DeepfakeAudioDataset(Dataset):
    
    
    def __init__(self, data_dir, split='train', sample_rate=16000, 
                 n_mels=80, max_length=None, augment=False):
        """
        Args:
            data_dir: Path to data directory (e.g., 'data/')
            split: 'train' or 'val'
            sample_rate: Target sample rate
            n_mels: Number of mel bands
            max_length: Maximum audio length in seconds (None = no limit)
            augment: Whether to apply data augmentation
        """
        self.data_dir = Path(data_dir) / split
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_length = max_length
        self.augment = augment
        
        # Load file paths and labels
        self.audio_files = []
        self.labels = []
        
        # Load real audio files (label = 0)
        real_dir = self.data_dir / 'real'
        if real_dir.exists():
            for audio_file in real_dir.glob('*.wav'):
                self.audio_files.append(str(audio_file))
                self.labels.append(0)  # Real = 0
            
            # Also check for other formats
            for ext in ['*.mp3', '*.flac', '*.ogg']:
                for audio_file in real_dir.glob(ext):
                    self.audio_files.append(str(audio_file))
                    self.labels.append(0)
        
        # Load fake audio files (label = 1)
        fake_dir = self.data_dir / 'fake'
        if fake_dir.exists():
            for audio_file in fake_dir.glob('*.wav'):
                self.audio_files.append(str(audio_file))
                self.labels.append(1)  # Fake = 1
            
            # Also check for other formats
            for ext in ['*.mp3', '*.flac', '*.ogg']:
                for audio_file in fake_dir.glob(ext):
                    self.audio_files.append(str(audio_file))
                    self.labels.append(1)
        
        print(f"Loaded {len(self.audio_files)} audio files from {split} split")
        print(f"  - Real: {self.labels.count(0)}")
        print(f"  - Fake: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.audio_files)
    
    def load_audio(self, audio_path):
        """Load and preprocess audio"""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        # Normalize
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        
        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Limit length if specified
        if self.max_length is not None:
            max_samples = int(self.sample_rate * self.max_length)
            if len(audio) > max_samples:
                # Random crop during training, center crop during validation
                if self.augment:
                    start = np.random.randint(0, len(audio) - max_samples)
                    audio = audio[start:start + max_samples]
                else:
                    start = (len(audio) - max_samples) // 2
                    audio = audio[start:start + max_samples]
            elif len(audio) < max_samples:
                # Pad if too short
                audio = np.pad(audio, (0, max_samples - len(audio)))
        
        # Ensure minimum length (0.5 seconds)
        min_samples = int(self.sample_rate * 0.5)
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)))
        
        return audio
    
    def augment_audio(self, audio):
        """Apply data augmentation"""
        if not self.augment:
            return audio
        
        # Random gain (volume change)
        if np.random.random() < 0.5:
            gain = np.random.uniform(0.7, 1.3)
            audio = audio * gain
            audio = np.clip(audio, -1.0, 1.0)
        
        # Add slight noise
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.005, audio.shape)
            audio = audio + noise
            audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    def create_mel_spectrogram(self, audio):
        """Convert audio to mel-spectrogram"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=512,
            hop_length=160,
            win_length=400
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Transpose to (time, frequency)
        mel_spec_db = mel_spec_db.T
        
        return mel_spec_db
    
    def __getitem__(self, idx):
        """Get one sample"""
        # Load audio
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        
        try:
            audio = self.load_audio(audio_path)
            
            # Apply augmentation
            audio = self.augment_audio(audio)
            
            # Create mel-spectrogram
            mel_spec = self.create_mel_spectrogram(audio)
            
            # Convert to tensors
            mel_tensor = torch.FloatTensor(mel_spec)
            audio_tensor = torch.FloatTensor(audio)
            label_tensor = torch.FloatTensor([label])
            
            return {
                'mel_spectrogram': mel_tensor,
                'raw_audio': audio_tensor,
                'label': label_tensor,
                'audio_path': audio_path
            }
        
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return a dummy sample
            mel_tensor = torch.zeros(100, self.n_mels)
            audio_tensor = torch.zeros(self.sample_rate)
            label_tensor = torch.FloatTensor([label])
            return {
                'mel_spectrogram': mel_tensor,
                'raw_audio': audio_tensor,
                'label': label_tensor,
                'audio_path': audio_path
            }


def create_dataloaders(data_dir, batch_size=16, num_workers=4, max_length=5.0):
    """
    Create train and validation dataloaders
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        max_length: Maximum audio length in seconds
    
    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = DeepfakeAudioDataset(
        data_dir=data_dir,
        split='train',
        sample_rate=16000,
        n_mels=80,
        max_length=max_length,
        augment=True  # Enable augmentation for training
    )
    
    val_dataset = DeepfakeAudioDataset(
        data_dir=data_dir,
        split='val',
        sample_rate=16000,
        n_mels=80,
        max_length=max_length,
        augment=False  # No augmentation for validation
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# TEST THE CODE


if __name__ == "__main__":
    print("Testing Dataset...")
    print("=" * 60)
    
    # Test with dummy data (you'll need to create this structure)
    data_dir = "data/"
    
    if not os.path.exists(data_dir):
        print(f"Please create the following structure:")
        print(f"{data_dir}")
        print(f"├── train/")
        print(f"│   ├── real/    (put real audio files here)")
        print(f"│   └── fake/    (put fake audio files here)")
        print(f"└── val/")
        print(f"    ├── real/")
        print(f"    └── fake/")
    else:
        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            data_dir=data_dir,
            batch_size=4,
            num_workers=0,  # Use 0 for debugging
            max_length=5.0
        )
        
        # Test loading a batch
        if len(train_loader) > 0:
            batch = next(iter(train_loader))
            
            print(f"\nBatch contents:")
            print(f"  Mel-spectrogram shape: {batch['mel_spectrogram'].shape}")
            print(f"  Raw audio shape: {batch['raw_audio'].shape}")
            print(f"  Labels shape: {batch['label'].shape}")
            print(f"  Labels: {batch['label'].squeeze().numpy()}")
            
            print("\n" + "=" * 60)
            print("Test passed! ✓")
        else:
            print("\nNo data found. Please add audio files to the data folders.")
