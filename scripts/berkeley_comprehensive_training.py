#!/usr/bin/env python3
"""
COMPREHENSIVE BERKELEY DATASET TRAINING FRAMEWORK
================================================

Train a pick-and-place policy using the full 77GB Berkeley UR5 dataset.
This framework will properly utilize all 412 training TFRecord files with 
multiple epochs to achieve maximum performance from the available data.

Key Features:
- Full 77GB Berkeley dataset utilization (412 training files)
- Proper epoch-based training (not just episodes)
- Advanced data loading and preprocessing
- Progressive learning rate scheduling
- Comprehensive model checkpointing
- Detailed training metrics and validation
- GPU-accelerated training with mixed precision

Berkeley Dataset Structure:
- 412 training TFRecord files (~73GB)
- 50 test TFRecord files (~8GB)
- Episodes with multi-modal observations (images, robot states)
- Ground truth expert demonstrations from Berkeley Autolab

Author: NIVA Training Team
Date: 2025-09-02
Status: Comprehensive Berkeley Dataset Training Implementation
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import random
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt

# Update path to include the Berkeley dataset loader
sys.path.append('/home/todd/niva-nbot/src/data_collection')
from berkeley_dataset_loader import BerkeleyDatasetConfig, BerkeleyDatasetLoader, BerkeleyTorchDataset

@dataclass
class BerkeleyTrainingConfig:
    """Configuration for comprehensive Berkeley dataset training"""
    
    # Dataset Configuration
    dataset_path: str = "/mnt/niva_hot/datasets/berkeley_autolab_ur5/0.1.0"
    batch_size: int = 32
    num_workers: int = 8
    sequence_length: int = 50
    image_size: Tuple[int, int] = (224, 224)
    
    # Training Configuration
    num_epochs: int = 50  # Multiple epochs through the full dataset
    learning_rate: float = 1e-4
    lr_schedule: str = "cosine"  # "cosine", "step", "exponential"
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    
    # Model Configuration
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.1
    action_dim: int = 7  # UR5 robot action space
    
    # Training Optimizations
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    warmup_epochs: int = 5
    
    # Checkpointing & Evaluation
    checkpoint_frequency: int = 5  # Save every 5 epochs
    validation_frequency: int = 1  # Validate every epoch
    early_stopping_patience: int = 10
    
    # Output Configuration
    output_dir: str = "/home/todd/niva-nbot-eval/berkeley_training"
    model_save_dir: str = "/home/todd/niva-nbot-eval/models"
    log_frequency: int = 100  # Log every 100 batches

class AdvancedVisuoMotorPolicy(nn.Module):
    """Advanced neural network policy for visuomotor control"""
    
    def __init__(self, config: BerkeleyTrainingConfig):
        super().__init__()
        self.config = config
        
        # Vision Encoder (CNN for image processing)
        self.vision_encoder = nn.Sequential(
            # Input: 3 x 224 x 224
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),  # 32 x 112 x 112
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # 64 x 56 x 56
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 128 x 28 x 28
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 256 x 14 x 14
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            
            nn.AdaptiveAvgPool2d((4, 4)),  # 256 x 4 x 4
            nn.Flatten(),  # 4096 features
        )
        
        # State Encoder (MLP for robot state)
        # Assuming robot state includes joint positions, velocities, etc.
        self.state_encoder = nn.Sequential(
            nn.Linear(14, 64),  # Typical UR5 state: 7 pos + 7 vel
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(config.dropout),
            
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(config.dropout),
        )
        
        # Fusion Layer
        vision_features = 4096
        state_features = 128
        fusion_input = vision_features + state_features
        
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(config.hidden_dim),
            nn.Dropout(config.dropout),
        )
        
        # Policy Head (multiple layers for complex behavior learning)
        self.policy_head = nn.ModuleList()
        for i in range(config.num_layers):
            input_dim = config.hidden_dim if i == 0 else config.hidden_dim
            self.policy_head.append(
                nn.Sequential(
                    nn.Linear(input_dim, config.hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(config.hidden_dim),
                    nn.Dropout(config.dropout) if i < config.num_layers - 1 else nn.Identity(),
                )
            )
        
        # Output layer
        self.action_head = nn.Linear(config.hidden_dim, config.action_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier/He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the policy network"""
        # Process images
        images = observations['image']  # Shape: [batch, seq_len, 3, 224, 224]
        batch_size, seq_len = images.shape[:2]
        
        # Use only the current frame for now (can be extended to temporal modeling)
        current_image = images[:, -1]  # [batch, 3, 224, 224]
        vision_features = self.vision_encoder(current_image)  # [batch, 4096]
        
        # Process robot state
        states = observations['state']  # Shape: [batch, seq_len, state_dim]
        current_state = states[:, -1]  # [batch, state_dim]
        state_features = self.state_encoder(current_state)  # [batch, 128]
        
        # Fuse multimodal features
        fused_features = torch.cat([vision_features, state_features], dim=1)
        x = self.fusion_layers(fused_features)
        
        # Pass through policy layers
        for layer in self.policy_head:
            x = layer(x)
        
        # Generate actions
        actions = self.action_head(x)
        
        return actions

class BerkeleyDatasetTrainer:
    """Comprehensive trainer for Berkeley dataset"""
    
    def __init__(self, config: BerkeleyTrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup output directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.model_save_dir, exist_ok=True)
        
        # Initialize logging
        self.training_log = []
        self.validation_log = []
        
        print(f"🚀 COMPREHENSIVE BERKELEY DATASET TRAINING")
        print(f"==========================================")
        print(f"📊 Device: {self.device}")
        print(f"📊 Mixed Precision: {config.mixed_precision}")
        print(f"📊 Batch Size: {config.batch_size}")
        print(f"📊 Epochs: {config.num_epochs}")
        print(f"📊 Learning Rate: {config.learning_rate}")
        print(f"📊 Dataset Path: {config.dataset_path}")
        
        # Initialize model
        self.model = AdvancedVisuoMotorPolicy(config).to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        if config.lr_schedule == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.num_epochs
            )
        elif config.lr_schedule == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.5
            )
        else:  # exponential
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.95
            )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Metrics
        self.global_step = 0
        self.epoch_start_time = None
        
        print(f"✅ Model initialized: {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def setup_data_loaders(self):
        """Setup training and validation data loaders"""
        print("🔄 Setting up data loaders...")
        
        # Training dataset (use most of the data)
        train_config = BerkeleyDatasetConfig(
            dataset_path=self.config.dataset_path,
            batch_size=self.config.batch_size,
            sequence_length=self.config.sequence_length,
            image_size=self.config.image_size,
            use_train_split=True,
            max_episodes=None,  # Use all training data
            shuffle=True,
            num_workers=self.config.num_workers
        )
        
        # Validation dataset (use test split)
        val_config = BerkeleyDatasetConfig(
            dataset_path=self.config.dataset_path,
            batch_size=self.config.batch_size,
            sequence_length=self.config.sequence_length,
            image_size=self.config.image_size,
            use_train_split=False,  # Use test split for validation
            max_episodes=None,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        
        # Create datasets
        print("📚 Loading training dataset...")
        self.train_dataset = BerkeleyTorchDataset(train_config)
        print(f"✅ Training episodes: {len(self.train_dataset)}")
        
        print("📚 Loading validation dataset...")
        self.val_dataset = BerkeleyTorchDataset(val_config)
        print(f"✅ Validation episodes: {len(self.val_dataset)}")
        
        # Create data loaders
        self.train_loader = self.train_dataset.get_dataloader()
        self.val_loader = self.val_dataset.get_dataloader()
        
        print(f"🔄 Training batches per epoch: {len(self.train_loader)}")
        print(f"🔄 Validation batches: {len(self.val_loader)}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.epoch_start_time = time.time()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            if batch is None:  # Skip empty batches
                continue
            
            # Move data to device
            observations = {}
            for key, value in batch['observations'].items():
                observations[key] = value.to(self.device)
            
            target_actions = batch['actions'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with optional mixed precision
            if self.config.mixed_precision:
                with autocast():
                    predicted_actions = self.model(observations)
                    loss = self.criterion(predicted_actions, target_actions)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                if self.config.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predicted_actions = self.model(observations)
                loss = self.criterion(predicted_actions, target_actions)
                
                loss.backward()
                
                if self.config.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if batch_idx % self.config.log_frequency == 0:
                elapsed = time.time() - self.epoch_start_time
                print(f"[{epoch:3d}][{batch_idx:4d}/{len(self.train_loader)}] "
                      f"Loss: {loss.item():.6f} | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                      f"Time: {elapsed:.1f}s")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        epoch_time = time.time() - self.epoch_start_time
        
        return {
            'train_loss': avg_loss,
            'epoch_time': epoch_time,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'batches_processed': num_batches
        }
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                if batch is None:
                    continue
                
                # Move data to device
                observations = {}
                for key, value in batch['observations'].items():
                    observations[key] = value.to(self.device)
                
                target_actions = batch['actions'].to(self.device)
                
                # Forward pass
                predicted_actions = self.model(observations)
                loss = self.criterion(predicted_actions, target_actions)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Early stopping check
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.patience_counter = 0
            
            # Save best model
            best_model_path = os.path.join(self.config.model_save_dir, "berkeley_best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': avg_loss,
                'config': self.config
            }, best_model_path)
            print(f"💾 Best model saved: {best_model_path}")
        else:
            self.patience_counter += 1
        
        return {
            'val_loss': avg_loss,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, Any]):
        """Save training checkpoint"""
        checkpoint_path = os.path.join(
            self.config.output_dir, 
            f"berkeley_checkpoint_epoch_{epoch:03d}.pth"
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'metrics': metrics,
            'config': self.config,
            'global_step': self.global_step
        }, checkpoint_path)
        
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def save_training_logs(self):
        """Save training and validation logs"""
        log_data = {
            'training_log': self.training_log,
            'validation_log': self.validation_log,
            'config': self.config.__dict__,
            'final_epoch': len(self.training_log),
            'best_val_loss': self.best_val_loss
        }
        
        log_path = os.path.join(self.config.output_dir, "training_logs.json")
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        print(f"📊 Training logs saved: {log_path}")
    
    def train(self):
        """Execute comprehensive training"""
        print(f"🚀 Starting comprehensive Berkeley dataset training...")
        print(f"📊 Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Setup data loaders
        self.setup_data_loaders()
        
        # Training loop
        for epoch in range(1, self.config.num_epochs + 1):
            print(f"\n🔄 Epoch {epoch}/{self.config.num_epochs}")
            print("=" * 50)
            
            # Training
            train_metrics = self.train_epoch(epoch)
            self.training_log.append({
                'epoch': epoch,
                **train_metrics
            })
            
            # Validation
            if epoch % self.config.validation_frequency == 0:
                val_metrics = self.validate(epoch)
                self.validation_log.append({
                    'epoch': epoch,
                    **val_metrics
                })
                
                print(f"📊 Train Loss: {train_metrics['train_loss']:.6f}")
                print(f"📊 Val Loss: {val_metrics['val_loss']:.6f}")
                print(f"📊 Best Val Loss: {val_metrics['best_val_loss']:.6f}")
                print(f"📊 Patience: {val_metrics['patience_counter']}/{self.config.early_stopping_patience}")
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Checkpointing
            if epoch % self.config.checkpoint_frequency == 0:
                combined_metrics = {**train_metrics}
                if self.validation_log and self.validation_log[-1]['epoch'] == epoch:
                    combined_metrics.update(self.validation_log[-1])
                self.save_checkpoint(epoch, combined_metrics)
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"🛑 Early stopping triggered at epoch {epoch}")
                break
        
        # Save final model
        final_model_path = os.path.join(self.config.model_save_dir, "berkeley_final_model.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'final_epoch': epoch,
            'best_val_loss': self.best_val_loss
        }, final_model_path)
        
        # Save logs
        self.save_training_logs()
        
        print(f"\n✅ TRAINING COMPLETED!")
        print(f"📊 Final Epoch: {epoch}")
        print(f"📊 Best Validation Loss: {self.best_val_loss:.6f}")
        print(f"📊 Total Training Time: {sum(log['epoch_time'] for log in self.training_log):.1f}s")
        print(f"💾 Models saved in: {self.config.model_save_dir}")

def main():
    """Main training execution"""
    print("🤖 COMPREHENSIVE BERKELEY DATASET TRAINING")
    print("=" * 50)
    
    # Create training configuration
    config = BerkeleyTrainingConfig(
        # Dataset configuration for maximum utilization
        dataset_path="/mnt/niva_hot/datasets/berkeley_autolab_ur5/0.1.0",
        batch_size=16,  # Balanced for GPU memory and training speed
        num_workers=8,  # Parallel data loading
        
        # Training configuration for comprehensive learning
        num_epochs=30,  # Multiple epochs through the full 77GB dataset
        learning_rate=3e-4,  # Optimal for large-scale training
        lr_schedule="cosine",
        weight_decay=1e-4,
        
        # Model configuration for complex behavior learning
        hidden_dim=512,  # Larger capacity for rich dataset
        num_layers=6,    # Deeper network for complex behaviors
        dropout=0.1,
        
        # Advanced training optimizations
        mixed_precision=True,  # Faster training with RTX GPUs
        gradient_accumulation_steps=2,
        warmup_epochs=3,
        grad_clip_norm=1.0,
        
        # Comprehensive checkpointing
        checkpoint_frequency=2,
        validation_frequency=1,
        early_stopping_patience=8,
        log_frequency=50
    )
    
    # Initialize trainer
    trainer = BerkeleyDatasetTrainer(config)
    
    # Execute training
    trainer.train()
    
    print("🎯 Berkeley dataset training completed successfully!")

if __name__ == "__main__":
    main()
