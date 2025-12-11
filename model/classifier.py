"""
Simple Classifier - Beginner Version
This combines audio features and decides if audio is real or fake
"""

import torch
import torch.nn as nn



# Simple Classifier


class SimpleClassifier(nn.Module):
    """
    This is a simple neural network that decides if audio is fake.
    
    It takes two types of features:
    1. Audio features (from the audio encoder)
    2. Anomaly features (special handcrafted features)
    
    Then it combines them and outputs a score:
    - 0.0 = definitely REAL
    - 1.0 = definitely FAKE
    - 0.5 = uncertain
    """
    
    def __init__(self, audio_feature_size=256, anomaly_feature_size=40,
                 hidden_size=256, dropout_rate=0.3):
        super().__init__()
        
        # Store sizes
        self.audio_feature_size = audio_feature_size
        self.anomaly_feature_size = anomaly_feature_size
        
        # Calculate total input size
        total_input_size = audio_feature_size + anomaly_feature_size
        
        # First layer
        self.layer1 = nn.Linear(total_input_size, hidden_size)
        
        # Activation function (adds non-linearity)
        self.activation = nn.ReLU()
        
        # Dropout (prevents overfitting)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Second layer (output layer)
        self.layer2 = nn.Linear(hidden_size, 1)
        
        # Sigmoid to get probability between 0 and 1
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, audio_features, anomaly_features):
        """
        Make a prediction
        
        Input:
            audio_features: (batch_size, audio_feature_size)
            anomaly_features: (batch_size, anomaly_feature_size)
        
        Output:
            fake_score: (batch_size, 1) - probability of being fake
        """
        # Step 1: Combine the features
        combined_features = torch.cat([audio_features, anomaly_features], dim=1)
        
        # Step 2: First layer
        x = self.layer1(combined_features)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Step 3: Second layer
        x = self.layer2(x)
        
        # Step 4: Sigmoid to get probability
        fake_score = self.sigmoid(x)
        
        return fake_score



# Complete Detection Model


class DeepfakeDetector(nn.Module):
    """
    This is the complete deepfake detection model.
    
    It combines three components:
    1. Audio Encoder - analyzes mel-spectrogram
    2. Anomaly Extractor - extracts special features
    3. Classifier - makes final decision
    """
    
    def __init__(self, audio_encoder, anomaly_extractor,
                 audio_feature_size=256, anomaly_feature_size=40,
                 hidden_size=256, dropout_rate=0.3):
        super().__init__()
        
        # Store the three components
        self.audio_encoder = audio_encoder
        self.anomaly_extractor = anomaly_extractor
        
        # Create the classifier
        self.classifier = SimpleClassifier(
            audio_feature_size=audio_feature_size,
            anomaly_feature_size=anomaly_feature_size,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate
        )
    
    def forward(self, mel_spectrogram, raw_audio):
        """
        Process audio and make prediction
        
        Input:
            mel_spectrogram: (batch_size, time_steps, mel_bands)
            raw_audio: (batch_size, audio_length)
        
        Output:
            fake_score: (batch_size, 1) - probability of being fake
        """
        # Get audio features
        audio_features = self.audio_encoder(mel_spectrogram)
        
        # Get anomaly features
        anomaly_features = self.anomaly_extractor(raw_audio)
        
        # Make prediction
        fake_score = self.classifier(audio_features, anomaly_features)
        
        return fake_score
    
    def predict(self, mel_spectrogram, raw_audio, threshold=0.5):
        """
        Make prediction with label
        
        Returns:
            fake_scores: probability values
            labels: 0 for REAL, 1 for FAKE
        """
        # Get scores
        fake_scores = self.forward(mel_spectrogram, raw_audio)
        
        # Apply threshold to get labels
        labels = (fake_scores > threshold).float()
        
        return fake_scores, labels



# Deeper Classifier (More Layers)


class DeeperClassifier(nn.Module):
    """
    A more complex classifier with multiple layers.
    Use this if the simple classifier doesn't work well enough.
    """
    
    def __init__(self, audio_feature_size=256, anomaly_feature_size=40,
                 layer_sizes=[256, 128, 64], dropout_rate=0.3):
        super().__init__()
        
        # Calculate input size
        input_size = audio_feature_size + anomaly_feature_size
        
        # Build the layers
        layers = []
        current_size = input_size
        
        # Add hidden layers
        for layer_size in layer_sizes:
            layers.append(nn.Linear(current_size, layer_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_size = layer_size
        
        # Add output layer
        layers.append(nn.Linear(current_size, 1))
        layers.append(nn.Sigmoid())
        
        # Combine all layers
        self.network = nn.Sequential(*layers)
    
    def forward(self, audio_features, anomaly_features):
        """Make prediction"""
        # Combine features
        combined = torch.cat([audio_features, anomaly_features], dim=1)
        
        # Pass through network
        fake_score = self.network(combined)
        
        return fake_score



# TEST THE CODE


if __name__ == "__main__":
    print("Testing Classifier...")
    print("=" * 50)
    
    # Create simple classifier
    classifier = SimpleClassifier(
        audio_feature_size=256,
        anomaly_feature_size=40,
        hidden_size=256,
        dropout_rate=0.3
    )
    
    # Create fake features
    batch_size = 4
    audio_features = torch.randn(batch_size, 256)
    anomaly_features = torch.randn(batch_size, 40)
    
    print(f"Audio features shape: {audio_features.shape}")
    print(f"Anomaly features shape: {anomaly_features.shape}")
    
    # Make predictions
    fake_scores = classifier(audio_features, anomaly_features)
    print(f"Output shape: {fake_scores.shape}")
    print(f"Fake scores: {fake_scores.squeeze().detach().numpy()}")
    
    # Count parameters
    total_params = sum(p.numel() for p in classifier.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("=" * 50)
    print("Test passed! ✓")