"""
Simple Anomaly Feature Extractor - Beginner Version
This extracts special features from audio that help detect fakes
"""

import numpy as np
import librosa
import torch
import torch.nn as nn


# Calculate Statistics


def calculate_statistics(values):
    """
    Calculate basic statistics from a list of values
    Returns: [mean, std, min, max]
    """
    if len(values) == 0:
        return np.zeros(4)
    
    mean_value = np.mean(values)
    std_value = np.std(values)
    min_value = np.min(values)
    max_value = np.max(values)
    
    stats = np.array([mean_value, std_value, min_value, max_value])
    
    # Replace any NaN with 0
    stats = np.nan_to_num(stats, nan=0.0)
    
    return stats



# Extract Spectral Features (Frequency-based)


def get_spectral_centroid(audio, sample_rate):
    """
    Spectral centroid = "brightness" of sound
    Higher values = more high frequencies (brighter)
    Lower values = more low frequencies (darker)
    """
    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate,
        n_fft=2048,
        hop_length=512
    )
    return centroid.flatten()


def get_spectral_rolloff(audio, sample_rate):
    """
    Spectral rolloff = frequency where most energy is below
    Tells us about the shape of the sound spectrum
    """
    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sample_rate,
        n_fft=2048,
        hop_length=512,
        roll_percent=0.85
    )
    return rolloff.flatten()


def get_spectral_flatness(audio):
    """
    Spectral flatness = how noise-like the sound is
    Close to 1 = noise-like
    Close to 0 = tonal/musical
    """
    flatness = librosa.feature.spectral_flatness(
        y=audio,
        n_fft=2048,
        hop_length=512
    )
    return flatness.flatten()


def get_zero_crossing_rate(audio):
    """
    Zero-crossing rate = how often the signal changes sign
    High ZCR = noisy/high frequency
    Low ZCR = low frequency
    """
    zcr = librosa.feature.zero_crossing_rate(
        audio,
        frame_length=2048,
        hop_length=512
    )
    return zcr.flatten()



# Extract Prosody Features (Speech rhythm/melody)


def get_pitch(audio, sample_rate):
    """
    Pitch = fundamental frequency of the voice
    Captures the melody/intonation of speech
    """
    # Estimate pitch using librosa
    pitches, magnitudes = librosa.piptrack(
        y=audio,
        sr=sample_rate,
        fmin=50.0,
        fmax=400.0,
        n_fft=2048,
        hop_length=512
    )
    
    # Extract pitch at each time frame
    pitch_values = []
    for time_index in range(pitches.shape[1]):
        # Find the frequency with highest magnitude
        max_index = magnitudes[:, time_index].argmax()
        pitch = pitches[max_index, time_index]
        pitch_values.append(pitch)
    
    return np.array(pitch_values)


def get_energy(audio):
    """
    Energy = loudness of the audio over time
    Captures volume variations in speech
    """
    energy = librosa.feature.rms(
        y=audio,
        frame_length=2048,
        hop_length=512
    )
    return energy.flatten()


def get_formants(audio, sample_rate):
    """
    Formants = resonant frequencies in speech
    F1, F2, F3, F4 are different resonances
    These are unique to each speaker
    """
    # Calculate LPC order
    lpc_order = int(2 + sample_rate / 1000)
    
    try:
        # Get LPC coefficients
        lpc_coeffs = librosa.lpc(audio, order=lpc_order)
        
        # Find roots
        roots = np.roots(lpc_coeffs)
        
        # Keep only stable roots
        roots = roots[np.abs(roots) < 1]
        
        # Convert to frequencies
        angles = np.angle(roots)
        frequencies = angles * (sample_rate / (2 * np.pi))
        
        # Keep positive frequencies and sort
        frequencies = frequencies[frequencies > 0]
        frequencies = np.sort(frequencies)
        
        # Get first 4 formants
        formants = []
        for i in range(4):
            if i < len(frequencies):
                formants.append(frequencies[i])
            else:
                formants.append(0.0)
        
        return formants
        
    except:
        # If error, return zeros
        return [0.0, 0.0, 0.0, 0.0]



# Complete Anomaly Feature Extractor


class AnomalyFeatureExtractor(nn.Module):
    """
    This extracts special features from audio that help detect deepfakes.
    
    Features extracted:
    1. Spectral features (brightness, rolloff, flatness, zero-crossings)
    2. Prosody features (pitch, energy, formants)
    
    Total: 36 features
    """
    
    def __init__(self, sample_rate=16000, output_size=40):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.output_size = output_size
        
        # Layer to project features to desired output size
        self.projection = nn.Linear(36, output_size)
    
    def extract_all_features(self, audio):
        """
        Extract all features from audio
        Returns a 36-dimensional feature vector
        """
        all_features = []
        
        
        # SPECTRAL FEATURES (16 dimensions)
        
        
        # Spectral Centroid (4 stats)
        centroid = get_spectral_centroid(audio, self.sample_rate)
        all_features.extend(calculate_statistics(centroid))
        
        # Spectral Rolloff (4 stats)
        rolloff = get_spectral_rolloff(audio, self.sample_rate)
        all_features.extend(calculate_statistics(rolloff))
        
        # Spectral Flatness (4 stats)
        flatness = get_spectral_flatness(audio)
        all_features.extend(calculate_statistics(flatness))
        
        # Zero-Crossing Rate (4 stats)
        zcr = get_zero_crossing_rate(audio)
        all_features.extend(calculate_statistics(zcr))
        
        
        # PROSODY FEATURES (20 dimensions)
        
        
        # Pitch (4 stats)
        pitch = get_pitch(audio, self.sample_rate)
        all_features.extend(calculate_statistics(pitch))
        
        # Energy (4 stats)
        energy = get_energy(audio)
        all_features.extend(calculate_statistics(energy))
        
        # Formants (4 values)
        formants = get_formants(audio, self.sample_rate)
        all_features.extend(formants)
        
        # Pitch change rate (4 stats)
        if len(pitch) > 1:
            pitch_delta = np.diff(pitch)
            all_features.extend(calculate_statistics(pitch_delta))
        else:
            all_features.extend([0.0] * 4)
        
        # Energy change rate (4 stats)
        if len(energy) > 1:
            energy_delta = np.diff(energy)
            all_features.extend(calculate_statistics(energy_delta))
        else:
            all_features.extend([0.0] * 4)
        
        # Total: 16 + 20 = 36 features
        return np.array(all_features, dtype=np.float32)
    
    def forward(self, audio_batch):
        """
        Process a batch of audio
        
        Input: (batch_size, audio_length)
        Output: (batch_size, output_size)
        """
        batch_size = audio_batch.shape[0]
        device = audio_batch.device
        
        # Process each audio in the batch
        feature_list = []
        for i in range(batch_size):
            # Convert to numpy
            audio_numpy = audio_batch[i].detach().cpu().numpy()
            
            # Extract features
            features = self.extract_all_features(audio_numpy)
            feature_list.append(features)
        
        # Stack into batch
        features_batch = np.stack(feature_list, axis=0)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features_batch).to(device)
        
        # Project to output size
        output_features = self.projection(features_tensor)
        
        return output_features



# TEST THE CODE


if __name__ == "__main__":
    print("Testing Anomaly Feature Extractor...")
    print("=" * 50)
    
    # Create the extractor
    extractor = AnomalyFeatureExtractor(
        sample_rate=16000,
        output_size=40
    )
    
    # Create fake audio (2 seconds)
    batch_size = 4
    audio_length = 16000 * 2  # 2 seconds at 16kHz
    
    fake_audio = torch.randn(batch_size, audio_length)
    print(f"Input shape: {fake_audio.shape}")
    
    # Extract features
    features = extractor(fake_audio)
    print(f"Output shape: {features.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in extractor.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("=" * 50)
    print("Test passed! ✓")