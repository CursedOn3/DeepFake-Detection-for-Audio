"""
Configuration loader for Deepfake Audio Detection
Loads settings from YAML file
"""

import yaml
from pathlib import Path


class Config:
    """Configuration class"""
    
    def __init__(self, config_path='configs/config.yaml'):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def __getitem__(self, key):
        """Allow dictionary-style access"""
        return self.config[key]
    
    def get(self, key, default=None):
        """Get value with default"""
        return self.config.get(key, default)
    
    def to_dict(self):
        """Convert to dictionary"""
        return self.config


# Quick access functions

def load_config(config_path='configs/config.yaml'):
    """Load configuration from file"""
    return Config(config_path)


def get_default_config():
    """Get default configuration as dictionary"""
    return {
        # Data
        'data_dir': 'data/',
        'sample_rate': 16000,
        'n_mels': 80,
        'max_audio_length': 5.0,
        
        # Training
        'batch_size': 16,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_workers': 4,
        
        # Model
        'cnn_size': 256,
        'num_cnn_layers': 2,
        'embedding_size': 256,
        'num_heads': 8,
        'num_transformer_layers': 4,
        'feedforward_size': 1024,
        'dropout_rate': 0.1,
        'anomaly_feature_size': 40,
        'hidden_size': 256,
        
        # Inference
        'threshold': 0.5,
        
        # Paths
        'output_dir': 'outputs/'
    }


if __name__ == "__main__":
    # Test loading config
    try:
        config = load_config()
        print("Configuration loaded successfully!")
        print("\nData settings:")
        print(f"  Sample rate: {config['data']['sample_rate']}")
        print(f"  Mel bands: {config['data']['n_mels']}")
        print(f"  Max length: {config['data']['max_audio_length']}s")
        
        print("\nModel settings:")
        print(f"  Embedding size: {config['model']['audio_encoder']['embedding_size']}")
        print(f"  Transformer layers: {config['model']['audio_encoder']['num_transformer_layers']}")
        print(f"  Attention heads: {config['model']['audio_encoder']['num_heads']}")
        
        print("\nTraining settings:")
        print(f"  Batch size: {config['training']['batch_size']}")
        print(f"  Learning rate: {config['training']['learning_rate']}")
        print(f"  Num epochs: {config['training']['num_epochs']}")
        
    except FileNotFoundError:
        print("Config file not found. Using default configuration.")
        config = get_default_config()
        print(config)
