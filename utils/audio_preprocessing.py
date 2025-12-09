"""
handling loading, resampling, normalization, converting to spectrograms, framing
"""

import librosa
import numpy as np

#loading audio file

def load_audio(
        audio_path: str,
    target_sr: int = 16000
):
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    return audio, sr

# resampling audio file to 16 kHz
def resample_audio(
    audio: np.ndarray,
    original_sr: int,
    target_sr: int = 16000
) -> np.ndarray:
    if original_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
    return audio

# normalizing audio to [-1, 1] range
def normalize_audio(audio: np.ndarray) -> np.ndarray:
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    return audio

# converting audio to spectrogram
def audio_to_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400
) -> np.ndarray:
    spectrogram = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)
    return spectrogram_db

