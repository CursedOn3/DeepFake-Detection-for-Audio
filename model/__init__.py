"""
ERF-BA-TFD+ Audio-Only Deepfake Detection Model

This package contains the model components:
- audio_encoder: Transformer-based audio feature encoder
- anomaly_extractor: Spectral and prosody anomaly features
- classifier: Feature fusion and classification head
"""

from .audio_encoder import AudioEncoder
from .anomaly_extractor import AnomalyFeatureExtractor
from .classifier import DeepfakeClassifier, DeepfakeDetectionModel

__all__ = [
    'AudioEncoder',
    'AnomalyFeatureExtractor', 
    'DeepfakeClassifier',
    'DeepfakeDetectionModel'
]
