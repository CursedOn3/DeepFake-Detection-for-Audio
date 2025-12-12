"""
Simple Audio Encoder - Beginner Version
This uses Transformer to understand audio patterns
"""

import torch
import torch.nn as nn
import math


# Positional Encoding


class PositionalEncoding(nn.Module):
    """
    Adds position information to the audio features.
    
    Why? Transformers don't naturally understand "order" or "time",
    so we add special numbers to tell it which part comes first.
    """
    
    def __init__(self, embedding_size, max_length=5000):
        super().__init__()
        
        # Create position encodings
        position = torch.arange(max_length).unsqueeze(1)  # (max_length, 1)
        
        # Calculate the division term for sinusoids
        div_term = torch.exp(
            torch.arange(0, embedding_size, 2) * (-math.log(10000.0) / embedding_size)
        )
        
        # Create the positional encoding matrix
        pos_encoding = torch.zeros(max_length, embedding_size)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)  # Even positions
        pos_encoding[:, 1::2] = torch.cos(position * div_term)  # Odd positions
        
        # Add batch dimension and register as buffer
        pos_encoding = pos_encoding.unsqueeze(0)  # (1, max_length, embedding_size)
        self.register_buffer('pos_encoding', pos_encoding)
    
    def forward(self, x):
        """
        Add positional encoding to input
        
        Input: x with shape (batch_size, sequence_length, embedding_size)
        Output: x with position info added
        """
        # Add positional encoding (only for the length we need)
        x = x + self.pos_encoding[:, :x.size(1), :]
        return x



# CNN Feature Extractor


class SimpleCNN(nn.Module):
    """
    Convolutional layers to extract basic audio features.
    
    CNNs are good at finding patterns in nearby data points,
    like edges in images or phonemes in audio.
    """
    
    def __init__(self, input_size, output_size, num_layers=2):
        super().__init__()
        
        # Build CNN layers
        layers = []
        current_size = input_size
        
        for i in range(num_layers):
            # Each layer learns patterns at different scales
            layers.append(
                nn.Conv1d(
                    in_channels=current_size,
                    out_channels=output_size,
                    kernel_size=3,
                    padding=1
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(output_size))
            current_size = output_size
        
        self.cnn_layers = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Extract features using CNN
        
        Input: (batch_size, time_steps, mel_bands)
        Output: (batch_size, time_steps, output_size)
        """
        # Conv1d expects (batch_size, channels, time)
        x = x.transpose(1, 2)  # (batch, mel_bands, time)
        x = self.cnn_layers(x)
        x = x.transpose(1, 2)  # (batch, time, output_size)
        return x



# Transformer Encoder Block


class TransformerBlock(nn.Module):
    """
    One Transformer block with:
    1. Multi-head attention (looks at relationships between different parts)
    2. Feed-forward network (processes each position)
    3. Residual connections and normalization (helps training)
    """
    
    def __init__(self, embedding_size, num_heads, feedforward_size, dropout_rate):
        super().__init__()
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True  # Makes it easier to use
        )
        
        # Feed-forward network
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_size, feedforward_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feedforward_size, embedding_size)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        """
        Process input through the Transformer block
        
        Input: (batch_size, sequence_length, embedding_size)
        Output: same shape as input
        """
        # Multi-head attention with residual connection
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attended))
        
        # Feed-forward with residual connection
        fed_forward = self.feedforward(x)
        x = self.norm2(x + self.dropout(fed_forward))
        
        return x



# Complete Audio Encoder


class AudioEncoder(nn.Module):
    """
    Complete audio encoder that:
    1. Uses CNN to extract low-level features
    2. Adds positional information
    3. Uses Transformer to understand temporal patterns
    4. Pools features into a fixed-size vector
    
    This is the main component that "understands" the audio.
    """
    
    def __init__(self, mel_bands=80, cnn_size=256, num_cnn_layers=2,
                 embedding_size=256, num_heads=8, num_transformer_layers=4,
                 feedforward_size=1024, dropout_rate=0.1):
        super().__init__()
        
        # Store parameters
        self.mel_bands = mel_bands
        self.embedding_size = embedding_size
        
        # CNN for low-level feature extraction
        self.cnn = SimpleCNN(
            input_size=mel_bands,
            output_size=cnn_size,
            num_layers=num_cnn_layers
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            embedding_size=cnn_size,
            max_length=5000
        )
        
        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embedding_size=cnn_size,
                num_heads=num_heads,
                feedforward_size=feedforward_size,
                dropout_rate=dropout_rate
            )
            for _ in range(num_transformer_layers)
        ])
        
        # Final projection to output size
        if cnn_size != embedding_size:
            self.output_projection = nn.Linear(cnn_size, embedding_size)
        else:
            self.output_projection = None
    
    def forward(self, mel_spectrogram):
        """
        Encode audio into a feature vector
        
        Input:
            mel_spectrogram: (batch_size, time_steps, mel_bands)
        
        Output:
            audio_features: (batch_size, embedding_size)
        """
        # Extract CNN features
        x = self.cnn(mel_spectrogram)  # (batch, time, cnn_size)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Process through Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Pool across time dimension (mean pooling)
        # This gives us a fixed-size vector regardless of audio length
        x = torch.mean(x, dim=1)  # (batch, cnn_size)
        
        # Project to final embedding size if needed
        if self.output_projection is not None:
            x = self.output_projection(x)
        
        return x
    
    def get_sequence_features(self, mel_spectrogram):
        """
        Get features for each time step (useful for analysis)
        
        Returns: (batch_size, time_steps, embedding_size)
        """
        x = self.cnn(mel_spectrogram)
        x = self.pos_encoder(x)
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        if self.output_projection is not None:
            x = self.output_projection(x)
        
        return x



# TEST THE CODE


if __name__ == "__main__":
    print("Testing Audio Encoder...")
    print("=" * 50)
    
    # Create audio encoder
    encoder = AudioEncoder(
        mel_bands=80,
        cnn_size=256,
        num_cnn_layers=2,
        embedding_size=256,
        num_heads=8,
        num_transformer_layers=4,
        feedforward_size=1024,
        dropout_rate=0.1
    )
    
    # Create fake mel-spectrogram
    batch_size = 4
    time_steps = 100
    mel_bands = 80
    
    fake_mel = torch.randn(batch_size, time_steps, mel_bands)
    print(f"Input shape: {fake_mel.shape}")
    
    # Encode
    features = encoder(fake_mel)
    print(f"Output shape: {features.shape}")
    print(f"Output values (first 5): {features[0, :5].detach().numpy()}")
    
    # Test sequence features
    sequence_features = encoder.get_sequence_features(fake_mel)
    print(f"Sequence features shape: {sequence_features.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("=" * 50)
    print("Test passed! ✓")
