"""
Audio preprocessing utilities for deepfake detection.
Handles loading, resampling, normalization, converting to spectrograms, framing, and VAD.
"""

import librosa
import numpy as np
import torch
from typing import Tuple, Optional



# LOADING AUDIO


def load_audio(
    audio_path: str,
    target_sr: int = 16000
) -> Tuple[np.ndarray, int]:
    """
    Load audio file and convert to mono.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate in Hz (default: 16000)
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    return audio, sr



# RESAMPLING


def resample_audio(
    audio: np.ndarray,
    original_sr: int,
    target_sr: int = 16000
) -> np.ndarray:
    """
    Resample audio to target sample rate.
    
    Args:
        audio: Input audio signal
        original_sr: Original sample rate
        target_sr: Target sample rate (default: 16000)
        
    Returns:
        Resampled audio
    """
    if original_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
    return audio



# NORMALIZATION


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio amplitude to [-1, 1] range.
    
    Args:
        audio: Input audio signal
        
    Returns:
        Normalized audio signal
    """
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    return audio



# SPECTROGRAM CONVERSION


def audio_to_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400
) -> np.ndarray:
    """
    Convert audio to spectrogram (dB scale).
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate in Hz
        n_fft: FFT size
        hop_length: Hop length in samples
        win_length: Window length in samples
        
    Returns:
        Spectrogram in dB scale
    """
    spectrogram = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)
    return spectrogram_db


def audio_to_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_mels: int = 80,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400
) -> np.ndarray:
    """
    Convert audio to mel-spectrogram (dB scale).
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate in Hz
        n_mels: Number of mel bands (default: 80)
        n_fft: FFT size
        hop_length: Hop length in samples
        win_length: Window length in samples
        
    Returns:
        Mel-spectrogram in dB scale of shape (n_mels, time_steps)
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db



# FRAMING


def frame_audio(
    audio: np.ndarray,
    sample_rate: int,
    window_length_ms: float = 25.0,
    hop_length_ms: float = 10.0
) -> np.ndarray:
    """
    Frame audio signal into overlapping windows.
    
    Uses 25ms window length and 10ms hop size as per ERF-BA-TFD+ specification.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate in Hz
        window_length_ms: Window length in milliseconds (default: 25ms)
        hop_length_ms: Hop size in milliseconds (default: 10ms)
        
    Returns:
        Framed audio of shape (num_frames, frame_length)
    """
    # Convert ms to samples
    frame_length = int(sample_rate * window_length_ms / 1000)
    hop_length = int(sample_rate * hop_length_ms / 1000)
    
    # Calculate number of frames
    num_frames = 1 + (len(audio) - frame_length) // hop_length
    
    # Handle edge case where audio is too short
    if num_frames <= 0:
        padded_audio = np.pad(audio, (0, frame_length - len(audio)), mode='constant')
        return padded_audio.reshape(1, -1)
    
    # Create frames
    frames = np.zeros((num_frames, frame_length))
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frames[i] = audio[start:end]
    
    return frames


def apply_window(
    frames: np.ndarray,
    window_type: str = 'hamming'
) -> np.ndarray:
    """
    Apply windowing function to frames.
    
    Args:
        frames: Framed audio of shape (num_frames, frame_length)
        window_type: Type of window ('hamming', 'hann', 'blackman')
        
    Returns:
        Windowed frames
    """
    frame_length = frames.shape[1]
    
    if window_type == 'hamming':
        window = np.hamming(frame_length)
    elif window_type == 'hann':
        window = np.hanning(frame_length)
    elif window_type == 'blackman':
        window = np.blackman(frame_length)
    else:
        window = np.ones(frame_length)
    
    windowed_frames = frames * window
    return windowed_frames



# VOICE ACTIVITY DETECTION (VAD)


def apply_vad(
    audio: np.ndarray,
    sample_rate: int,
    frame_length_ms: float = 25.0,
    hop_length_ms: float = 10.0,
    energy_threshold: float = 0.01
) -> np.ndarray:
    """
    Apply simple Voice Activity Detection (VAD) to trim silence.
    Uses energy-based threshold to detect voice activity.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate in Hz
        frame_length_ms: Frame length in milliseconds
        hop_length_ms: Hop length in milliseconds
        energy_threshold: Energy threshold for voice detection (0-1)
        
    Returns:
        Audio with silence trimmed
    """
    # Frame the audio
    frames = frame_audio(audio, sample_rate, frame_length_ms, hop_length_ms)
    
    # Calculate energy for each frame
    frame_energies = np.sum(frames ** 2, axis=1)
    
    # Normalize energies
    max_energy = frame_energies.max()
    if max_energy > 0:
        frame_energies = frame_energies / max_energy
    
    # Find frames with energy above threshold
    voice_frames = frame_energies > energy_threshold
    
    if not np.any(voice_frames):
        return audio
    
    # Find start and end of voice activity
    voice_indices = np.where(voice_frames)[0]
    start_frame = voice_indices[0]
    end_frame = voice_indices[-1] + 1
    
    # Convert frame indices to sample indices
    hop_length = int(sample_rate * hop_length_ms / 1000)
    frame_length = int(sample_rate * frame_length_ms / 1000)
    
    start_sample = start_frame * hop_length
    end_sample = end_frame * hop_length + frame_length
    
    # Trim audio
    trimmed_audio = audio[start_sample:min(end_sample, len(audio))]
    
    return trimmed_audio



# UTILITY FUNCTIONS


def pad_or_truncate(
    audio: np.ndarray,
    target_length: int
) -> np.ndarray:
    """
    Pad or truncate audio to fixed length.
    
    Args:
        audio: Input audio signal
        target_length: Target length in samples
        
    Returns:
        Audio of exact target length
    """
    if len(audio) > target_length:
        audio = audio[:target_length]
    elif len(audio) < target_length:
        padding = target_length - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant', constant_values=0)
    
    return audio


def get_audio_duration(audio: np.ndarray, sample_rate: int) -> float:
    """
    Get audio duration in seconds.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate in Hz
        
    Returns:
        Duration in seconds
    """
    return len(audio) / sample_rate



# COMPLETE PREPROCESSING PIPELINES


def preprocess_audio(
    audio_path: str,
    target_sr: int = 16000,
    normalize: bool = True,
    apply_vad_flag: bool = True,
    target_length: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """
    Complete preprocessing pipeline for audio.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (default: 16000)
        normalize: Whether to normalize amplitude
        apply_vad_flag: Whether to apply VAD
        target_length: Optional fixed length in samples
        
    Returns:
        Tuple of (preprocessed_audio, sample_rate)
    """
    # Step 1: Load audio
    audio, sr = load_audio(audio_path, target_sr)
    
    # Step 2: Normalize amplitude
    if normalize:
        audio = normalize_audio(audio)
    
    # Step 3: Apply VAD to trim silence
    if apply_vad_flag:
        audio = apply_vad(audio, sr)
    
    # Step 4: Pad or truncate to fixed length if specified
    if target_length is not None:
        audio = pad_or_truncate(audio, target_length)
    
    return audio, sr


def preprocess_for_model(
    audio_path: str,
    target_sr: int = 16000,
    use_mel_spec: bool = True,
    n_mels: int = 80,
    apply_vad_flag: bool = True
) -> torch.Tensor:
    """
    Complete preprocessing pipeline for model input.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate
        use_mel_spec: Whether to use mel-spectrogram (True) or framed audio (False)
        n_mels: Number of mel bands
        apply_vad_flag: Whether to apply VAD
        
    Returns:
        Preprocessed audio tensor ready for model
    """
    # Load and preprocess audio
    audio, sr = preprocess_audio(
        audio_path,
        target_sr=target_sr,
        normalize=True,
        apply_vad_flag=apply_vad_flag
    )
    
    # Extract features
    if use_mel_spec:
        features = audio_to_mel_spectrogram(audio, sr, n_mels=n_mels)
        # Transpose to (time_steps, n_mels) for Transformer input
        features = features.T
    else:
        features = frame_audio(audio, sr)
    
    # Convert to PyTorch tensor
    features_tensor = torch.FloatTensor(features)
    
    return features_tensor


def preprocess_for_anomaly_extractor(
    audio_path: str,
    target_sr: int = 16000,
    apply_vad_flag: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Preprocess audio for anomaly feature extraction.
    Returns raw audio waveform for spectral/prosody analysis.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate
        apply_vad_flag: Whether to apply VAD
        
    Returns:
        Tuple of (raw_audio, sample_rate)
    """
    audio, sr = preprocess_audio(
        audio_path,
        target_sr=target_sr,
        normalize=True,
        apply_vad_flag=apply_vad_flag
    )
    
    return audio, sr

