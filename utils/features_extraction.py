"""
Feature Extraction for Audio Deepfake Detection
================================================

This file extracts important features from audio that help us detect fake audio.

How it works:
1. Take raw audio waveform as input
2. Use CNN layers to find patterns in the audio (like weird sounds or glitches)
3. Use a Transformer to understand the order/sequence of these patterns
4. Combine all features into one vector using mean pooling

Think of it like this:
- CNN = "Looking at small pieces of audio to find suspicious parts"
- Transformer = "Understanding how those pieces connect over time"
- Mean Pooling = "Combining everything into one summary"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class FeatureExtractor(nn.Module):
    """
    Extracts features from audio for deepfake detection.
    
    This class does 3 main things:
    1. Applies CNN layers to find local patterns in audio
    2. Uses a Transformer to understand temporal relationships
    3. Averages all features into a single vector
    """
    
    def __init__(self, conv_channels, conv_kernel_sizes, conv_strides, transformer_model_name):
        """
        Set up the feature extractor.
        
        Args:
            conv_channels: List of output channels for each CNN layer
                          Example: [64, 128, 256] means 3 CNN layers
            conv_kernel_sizes: List of kernel sizes for each CNN layer
                              Example: [3, 3, 3] means kernel size of 3 for all
            conv_strides: List of stride values for each CNN layer
                         Example: [1, 1, 1] means stride of 1 for all
            transformer_model_name: Name of the pretrained transformer
                                   Example: "bert-base-uncased"
        """
        # Call parent class constructor (required for PyTorch modules)
        super(FeatureExtractor, self).__init__()
        
       
        # STEP 1: Create CNN Layers
       
        # CNN layers look at small windows of audio to find patterns
        # Like looking at individual puzzle pieces
        
        self.conv_layers = nn.ModuleList()  # This stores all our CNN layers
        
        # Start with 1 input channel (mono audio)
        in_channels = 1
        
        # Create each CNN layer one by one
        for i in range(len(conv_channels)):
            # Get settings for this layer
            out_channels = conv_channels[i]
            kernel_size = conv_kernel_sizes[i]
            stride = conv_strides[i]
            
            # Create the CNN layer
            conv_layer = nn.Conv1d(
                in_channels=in_channels,      # How many channels coming in
                out_channels=out_channels,    # How many channels going out
                kernel_size=kernel_size,      # Size of the sliding window
                stride=stride                 # How much to move the window each time
            )
            
            # Add this layer to our list
            self.conv_layers.append(conv_layer)
            
            # Output channels become input channels for next layer
            in_channels = out_channels
        
       
        # STEP 2: Create Transformer
       
        # Transformer understands the ORDER of patterns
        # Like understanding how puzzle pieces fit together
        
        self.transformer = AutoModel.from_pretrained(transformer_model_name)
        
    def forward(self, audio_waveform):
        """
        Extract features from audio.
        
        Args:
            audio_waveform: Raw audio signal
                           Shape: (batch_size, num_samples)
                           Example: (32, 16000) for 32 audio clips of 1 second each at 16kHz
        
        Returns:
            features: Extracted feature vector
                     Shape: (batch_size, hidden_dim)
                     This is what we use to classify real vs fake
        """
        
       
        # STEP 1: Prepare audio for CNN
       
        # CNN expects input shape: (batch_size, channels, length)
        # Our audio is: (batch_size, length)
        # So we need to add a channel dimension
        
        x = audio_waveform.unsqueeze(1)  # Add channel dimension
        # Now shape is: (batch_size, 1, num_samples)
        
       
        # STEP 2: Pass through CNN layers
       
        # Each CNN layer finds different patterns
        # ReLU activation adds non-linearity (helps model learn complex patterns)
        
        for conv_layer in self.conv_layers:
            x = conv_layer(x)     # Apply convolution
            x = F.relu(x)         # Apply ReLU activation
        # After CNN, shape is: (batch_size, out_channels, seq_length)
        
       
        # STEP 3: Prepare for Transformer
       
        # Transformer expects: (batch_size, seq_length, feature_dim)
        # CNN output is: (batch_size, feature_dim, seq_length)
        # So we need to swap the last two dimensions
        
        x = x.permute(0, 2, 1)  # Swap dimensions
        # Now shape is: (batch_size, seq_length, feature_dim)
        
       
        # STEP 4: Pass through Transformer
       
        # Transformer looks at all time steps and understands relationships
        
        transformer_output = self.transformer(inputs_embeds=x)
        x = transformer_output.last_hidden_state
        # Shape is still: (batch_size, seq_length, hidden_dim)
        
       
        # STEP 5: Mean Pooling
       
        # We have features for each time step, but we need ONE feature vector
        # Mean pooling = average all time steps together
        
        features = torch.mean(x, dim=1)  # Average over time dimension
        # Final shape: (batch_size, hidden_dim)
        
        return features


# ============================================
# EXAMPLE USAGE (for beginners)
# ============================================

def example_usage():
    """
    Example showing how to use the FeatureExtractor.
    Run this to understand how it works!
    """
    
    # Define CNN settings
    # We'll use 3 CNN layers with increasing channels
    conv_channels = [64, 128, 256]     # Output channels for each layer
    conv_kernel_sizes = [3, 3, 3]      # Kernel size of 3 for all layers
    conv_strides = [1, 1, 1]           # Stride of 1 for all layers
    
    # Choose a small transformer model
    transformer_name = "bert-base-uncased"
    
    # Create the feature extractor
    print("Creating FeatureExtractor...")
    extractor = FeatureExtractor(
        conv_channels=conv_channels,
        conv_kernel_sizes=conv_kernel_sizes,
        conv_strides=conv_strides,
        transformer_model_name=transformer_name
    )
    
    # Create fake audio data for testing
    batch_size = 2           # 2 audio clips
    num_samples = 16000      # 1 second of audio at 16kHz
    
    fake_audio = torch.randn(batch_size, num_samples)  # Random audio
    print(f"Input audio shape: {fake_audio.shape}")
    
    # Extract features
    print("Extracting features...")
    features = extractor(fake_audio)
    print(f"Output features shape: {features.shape}")
    
    print("Done! Features are ready for classification.")


# Run example if this file is run directly
if __name__ == "__main__":
    example_usage()