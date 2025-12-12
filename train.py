"""
Training Script for Deepfake Audio Detection
This trains the model to detect fake audio
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import json
from pathlib import Path

# Import our components
from model.audio_encoder import AudioEncoder
from model.anomaly_extractor import AnomalyFeatureExtractor
from model.classifier import DeepfakeDetector
from data_loader import create_dataloaders


class Trainer:
    """
    Handles the training process
    """
    
    def __init__(self, config):
        """
        Initialize trainer
        
        Args:
            config: Dictionary with training configuration
        """
        self.config = config
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create model
        print("\nBuilding model...")
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Create optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # For tracking
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # Create output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'logs'))
    
    def _build_model(self):
        """Build the complete model"""
        # Audio encoder
        audio_encoder = AudioEncoder(
            mel_bands=self.config['n_mels'],
            cnn_size=self.config['cnn_size'],
            num_cnn_layers=self.config['num_cnn_layers'],
            embedding_size=self.config['embedding_size'],
            num_heads=self.config['num_heads'],
            num_transformer_layers=self.config['num_transformer_layers'],
            feedforward_size=self.config['feedforward_size'],
            dropout_rate=self.config['dropout_rate']
        )
        
        # Anomaly extractor
        anomaly_extractor = AnomalyFeatureExtractor(
            sample_rate=self.config['sample_rate'],
            output_size=self.config['anomaly_feature_size']
        )
        
        # Complete model
        model = DeepfakeDetector(
            audio_encoder=audio_encoder,
            anomaly_extractor=anomaly_extractor,
            audio_feature_size=self.config['embedding_size'],
            anomaly_feature_size=self.config['anomaly_feature_size'],
            hidden_size=self.config['hidden_size'],
            dropout_rate=self.config['dropout_rate']
        )
        
        return model
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Get data
            mel_spec = batch['mel_spectrogram'].to(self.device)
            raw_audio = batch['raw_audio'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(mel_spec, raw_audio)
            
            # Calculate loss
            loss = self.criterion(predictions, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            
            # Calculate accuracy
            predicted = (predictions > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })
        
        # Calculate average metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        # Track predictions for more detailed metrics
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Get data
                mel_spec = batch['mel_spectrogram'].to(self.device)
                raw_audio = batch['raw_audio'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                predictions = self.model(mel_spec, raw_audio)
                
                # Calculate loss
                loss = self.criterion(predictions, labels)
                total_loss += loss.item()
                
                # Calculate accuracy
                predicted = (predictions > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                # Store for metrics
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, train_loader, val_loader, num_epochs):
        """Main training loop"""
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(is_best=is_best)
            
            # Early stopping check
            if self.optimizer.param_groups[0]['lr'] < 1e-7:
                print("\nLearning rate too small, stopping training...")
                break
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("="*60)
        
        self.writer.close()


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Deepfake Audio Detection Model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/',
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='outputs/',
                        help='Path to save checkpoints and logs')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--max_audio_length', type=float, default=5.0,
                        help='Maximum audio length in seconds')
    
    # Model arguments
    parser.add_argument('--embedding_size', type=int, default=256,
                        help='Embedding size')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_transformer_layers', type=int, default=4,
                        help='Number of Transformer layers')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create configuration dictionary
    config = {
        # Data
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'sample_rate': 16000,
        'n_mels': 80,
        'max_audio_length': args.max_audio_length,
        
        # Training
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'num_workers': args.num_workers,
        
        # Model
        'cnn_size': 256,
        'num_cnn_layers': 2,
        'embedding_size': args.embedding_size,
        'num_heads': args.num_heads,
        'num_transformer_layers': args.num_transformer_layers,
        'feedforward_size': 1024,
        'dropout_rate': 0.1,
        'anomaly_feature_size': 40,
        'hidden_size': 256
    }
    
    # Save configuration
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print("Configuration:")
    print(json.dumps(config, indent=2))
    
    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader = create_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        max_length=config['max_audio_length']
    )
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(train_loader, val_loader, config['num_epochs'])


if __name__ == "__main__":
    main()
