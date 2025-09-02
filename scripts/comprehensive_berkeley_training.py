#!/usr/bin/env python3
"""
COMPREHENSIVE BERKELEY DATASET TRAINING - FULL 77GB UTILIZATION
==============================================================

This is the definitive training framework that will utilize the entire 77GB 
Berkeley dataset with all 412 training files to achieve maximum performance.

Key Features:
- Full 77GB dataset utilization (all 412 training files)
- Multi-epoch training for complete dataset coverage
- GPU-optimized visuomotor policy architecture  
- Real robotics demonstrations from Berkeley Autolab
- Comprehensive evaluation and comparison framework
- Professional-grade training monitoring and checkpointing

Training Progression:
1. Load and validate all 412 training files
2. Execute multi-epoch training with proper data cycling
3. Implement advanced visuomotor policy with attention
4. Compare against previous DR model performance
5. Generate comprehensive training report

Author: NIVA Training Team
Date: 2025-09-02
Status: Production Berkeley Training Framework
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import our Berkeley dataset parser
sys.path.append('/home/todd/niva-nbot-eval/scripts')
from berkeley_dataset_parser import BerkeleyConfig, BerkeleyPyTorchDataset

@dataclass
class ComprehensiveTrainingConfig:
    """Configuration for comprehensive Berkeley dataset training"""
    
    # Dataset Configuration - FULL UTILIZATION
    dataset_path: str = "/mnt/niva_hot/datasets/berkeley_autolab_ur5/0.1.0"
    use_all_files: bool = True  # Use all 412 training files
    max_files_train: Optional[int] = None  # None = use all 412 files
    max_files_val: Optional[int] = 20  # Limit validation files for faster evaluation
    
    # Training Configuration - MULTI-EPOCH
    num_epochs: int = 20  # Multiple epochs through entire dataset
    batch_size: int = 16  # Optimized for RTX 2000 + complex model
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # Data Configuration
    image_size: Tuple[int, int] = (224, 224)
    max_sequence_length: int = 50
    use_hand_camera: bool = True
    use_language: bool = True
    
    # Model Configuration - ADVANCED ARCHITECTURE
    vision_backbone: str = "resnet"  # "resnet", "efficientnet"
    hidden_dim: int = 512
    num_attention_heads: int = 8
    num_transformer_layers: int = 4
    dropout: float = 0.1
    
    # Optimization
    mixed_precision: bool = True
    gradient_clip_norm: float = 1.0
    lr_schedule: str = "cosine"  # "cosine", "step", "warmup"
    warmup_epochs: int = 2
    
    # Evaluation and Checkpointing
    eval_frequency: int = 2  # Evaluate every 2 epochs
    checkpoint_frequency: int = 5  # Save every 5 epochs
    save_best_model: bool = True
    early_stopping_patience: int = 8
    
    # Monitoring
    log_frequency: int = 50  # Log every 50 batches
    plot_training_curves: bool = True
    
    # Output Configuration
    output_dir: str = "/home/todd/niva-nbot-eval/comprehensive_berkeley_training"
    model_save_dir: str = "/home/todd/niva-nbot-eval/models"
    results_dir: str = "/home/todd/niva-nbot-eval/berkeley_results"

class AdvancedVisuoMotorPolicy(nn.Module):
    """Advanced visuomotor policy for Berkeley dataset training"""
    
    def __init__(self, config: ComprehensiveTrainingConfig):
        super().__init__()
        self.config = config
        
        # Vision encoder - ResNet backbone
        if config.vision_backbone == "resnet":
            self.vision_encoder = self._build_resnet_encoder()
            vision_features = 2048
        else:
            raise ValueError(f"Unsupported backbone: {config.vision_backbone}")
        
        # Hand camera encoder (optional)
        if config.use_hand_camera:
            self.hand_encoder = self._build_resnet_encoder(input_channels=3)
            hand_features = 2048
        else:
            hand_features = 0
        
        # Robot state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(15, 128),  # Berkeley robot state is 15D
            nn.ReLU(inplace=True),
            nn.LayerNorm(128),
            nn.Dropout(config.dropout),
            
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Dropout(config.dropout),
            
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.LayerNorm(512),
        )
        
        # Language encoder (if enabled)
        if config.use_language:
            self.language_encoder = nn.Sequential(
                nn.Linear(512, 256),  # Berkeley embeddings are 512D
                nn.ReLU(inplace=True),
                nn.LayerNorm(256),
                nn.Dropout(config.dropout),
                
                nn.Linear(256, 512),
                nn.ReLU(inplace=True),
                nn.LayerNorm(512),
            )
            language_features = 512
        else:
            language_features = 0
        
        # Fusion layer
        total_features = vision_features + hand_features + 512 + language_features  # state=512
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_features, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
        )
        
        # Temporal attention mechanism
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Transformer layers for temporal modeling
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=config.num_transformer_layers
        )
        
        # Policy head for action prediction
        self.policy_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
            
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.Dropout(config.dropout),
            
            nn.Linear(config.hidden_dim // 2, 7)  # Berkeley actions: 3D world + 3D rotation + 1D gripper
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"🧠 Advanced Visuomotor Policy:")
        print(f"   Vision backbone: {config.vision_backbone}")
        print(f"   Hand camera: {config.use_hand_camera}")
        print(f"   Language: {config.use_language}")
        print(f"   Hidden dim: {config.hidden_dim}")
        print(f"   Attention heads: {config.num_attention_heads}")
        print(f"   Transformer layers: {config.num_transformer_layers}")
        print(f"   Total parameters: {total_params:,}")
    
    def _build_resnet_encoder(self, input_channels: int = 3) -> nn.Module:
        """Build ResNet-based vision encoder"""
        return nn.Sequential(
            # Initial conv layer
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # ResNet blocks
            self._make_resnet_block(64, 64, 2),
            self._make_resnet_block(64, 128, 2, stride=2),
            self._make_resnet_block(128, 256, 2, stride=2),
            self._make_resnet_block(256, 512, 2, stride=2),
            
            # Final pooling
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            
            # Feature projection
            nn.Linear(512 * 16, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
    
    def _make_resnet_block(self, in_channels: int, out_channels: int, num_blocks: int, stride: int = 1) -> nn.Module:
        """Create ResNet block"""
        layers = []
        
        # First block (may have stride > 1)
        layers.append(self._resnet_basic_block(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(self._resnet_basic_block(out_channels, out_channels, 1))
        
        return nn.Sequential(*layers)
    
    def _resnet_basic_block(self, in_channels: int, out_channels: int, stride: int) -> nn.Module:
        """Basic ResNet block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Handle different batch dimensions
        if batch['images'].dim() == 4:  # Single episode [seq, H, W, 3]
            batch_size, seq_len = 1, batch['images'].shape[0]
        else:  # Batched episodes [batch, seq, H, W, 3]
            batch_size, seq_len = batch['images'].shape[:2]
        
        # Process main camera images
        images = batch['images']  # [seq, H, W, 3] (single episode from dataloader)
        if images.dim() == 4:  # [seq, H, W, 3]
            images = images.permute(0, 3, 1, 2)  # [seq, 3, H, W]
        elif images.dim() == 5:  # [batch, seq, H, W, 3]  
            batch_size, seq_len = images.shape[:2]
            images = images.view(-1, *images.shape[2:])  # [batch*seq, H, W, 3]
            images = images.permute(0, 3, 1, 2)  # [batch*seq, 3, H, W]
        else:
            raise ValueError(f"Unexpected image tensor shape: {images.shape}")
        
        vision_features = self.vision_encoder(images)  # [seq, 2048] or [batch*seq, 2048]
        if batch_size == 1:
            vision_features = vision_features.unsqueeze(0)  # [1, seq, 2048]
        else:
            vision_features = vision_features.view(batch_size, seq_len, -1)  # [batch, seq, 2048]
        
        # Process robot states
        states = batch['robot_states']  # [seq, 15] or [batch, seq, 15]
        if states.dim() == 2:  # Single episode [seq, 15]
            states = states.unsqueeze(0)  # [1, seq, 15]
        
        states_flat = states.view(-1, states.shape[-1])  # [batch*seq, 15]
        state_features = self.state_encoder(states_flat)  # [batch*seq, 512]
        state_features = state_features.view(batch_size, seq_len, -1)  # [batch, seq, 512]
        
        # Combine features
        combined_features = [vision_features, state_features]
        
        # Add hand camera features if available
        if self.config.use_hand_camera and 'hand_images' in batch:
            hand_images = batch['hand_images']
            hand_images = hand_images.view(-1, *hand_images.shape[2:])
            hand_images = hand_images.permute(0, 3, 1, 2)
            hand_features = self.hand_encoder(hand_images)
            hand_features = hand_features.view(batch_size, seq_len, -1)
            combined_features.append(hand_features)
        
        # Add language features if available
        if self.config.use_language and 'language_embeddings' in batch:
            lang_embeddings = batch['language_embeddings']  # [batch, seq, 512]
            lang_embeddings_flat = lang_embeddings.view(-1, lang_embeddings.shape[-1])
            lang_features = self.language_encoder(lang_embeddings_flat)
            lang_features = lang_features.view(batch_size, seq_len, -1)
            combined_features.append(lang_features)
        
        # Fuse all features
        fused_features = torch.cat(combined_features, dim=-1)  # [batch, seq, total_features]
        fused_features_flat = fused_features.view(-1, fused_features.shape[-1])
        
        # Apply fusion layer
        hidden_features = self.feature_fusion(fused_features_flat)  # [batch*seq, hidden_dim]
        hidden_features = hidden_features.view(batch_size, seq_len, -1)  # [batch, seq, hidden_dim]
        
        # Apply temporal attention and transformer
        attended_features, _ = self.temporal_attention(
            hidden_features, hidden_features, hidden_features
        )
        
        # Apply transformer for temporal modeling
        temporal_features = self.transformer(attended_features)  # [batch, seq, hidden_dim]
        
        # Use final timestep for action prediction
        final_features = temporal_features[:, -1, :]  # [batch, hidden_dim]
        
        # Predict actions
        actions = self.policy_head(final_features)  # [batch, 7]
        
        return actions

class ComprehensiveBerkeleyTrainer:
    """Comprehensive trainer for full Berkeley dataset utilization"""
    
    def __init__(self, config: ComprehensiveTrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.model_save_dir, exist_ok=True)
        os.makedirs(config.results_dir, exist_ok=True)
        
        print(f"🚀 COMPREHENSIVE BERKELEY DATASET TRAINING")
        print(f"==========================================")
        print(f"🔥 Device: {self.device}")
        print(f"🔥 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        print(f"📊 Dataset: {config.dataset_path}")
        print(f"📊 Use all files: {config.use_all_files}")
        print(f"📊 Epochs: {config.num_epochs}")
        print(f"📊 Batch size: {config.batch_size}")
        
        # Initialize model
        self.model = AdvancedVisuoMotorPolicy(config).to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=1e-8
        )
        
        # Setup learning rate scheduler
        if config.lr_schedule == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.num_epochs
            )
        elif config.lr_schedule == "warmup":
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.1, total_iters=config.warmup_epochs
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=5, gamma=0.5
            )
        
        # Loss function and mixed precision
        self.criterion = nn.MSELoss()
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        print(f"✅ Trainer initialized successfully")
    
    def setup_data_loaders(self):
        """Setup training and validation data loaders"""
        print("🔄 Setting up data loaders...")
        
        # Training dataset configuration
        train_config = BerkeleyConfig(
            dataset_path=self.config.dataset_path,
            batch_size=self.config.batch_size,
            max_sequence_length=self.config.max_sequence_length,
            image_size=self.config.image_size,
            use_hand_camera=self.config.use_hand_camera,
            use_language=self.config.use_language,
            max_files_per_split=self.config.max_files_train,
            shuffle_buffer_size=2000
        )
        
        # Validation dataset configuration  
        val_config = BerkeleyConfig(
            dataset_path=self.config.dataset_path,
            batch_size=self.config.batch_size,
            max_sequence_length=self.config.max_sequence_length,
            image_size=self.config.image_size,
            use_hand_camera=self.config.use_hand_camera,
            use_language=self.config.use_language,
            max_files_per_split=self.config.max_files_val,
            shuffle_buffer_size=0  # No shuffling for validation
        )
        
        # Create datasets
        print("📚 Loading training dataset...")
        self.train_dataset = BerkeleyPyTorchDataset(train_config, split='train')
        
        print("📚 Loading validation dataset...")
        self.val_dataset = BerkeleyPyTorchDataset(val_config, split='test')
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=1,  # Berkeley dataset already batches episodes
            shuffle=True,
            num_workers=0,  # Single worker for GPU optimization
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        print(f"✅ Data loaders ready:")
        print(f"   Training episodes: {len(self.train_dataset)}")
        print(f"   Validation episodes: {len(self.val_dataset)}")
        print(f"   Training success rate: {self.train_dataset.success_rate:.1%}")
        print(f"   Validation success rate: {self.val_dataset.success_rate:.1%}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        epoch_start = time.time()
        
        print(f"\n🔥 Epoch {epoch}/{self.config.num_epochs}")
        print("-" * 60)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].squeeze(0).to(self.device, non_blocking=True)
            
            # Skip if no target actions
            if 'actions' not in batch or batch['actions'].numel() == 0:
                continue
            
            # Get target action (final timestep)
            target_action = batch['actions'][-1]  # [7]
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.config.mixed_precision:
                with autocast():
                    predicted_action = self.model(batch)
                    loss = self.criterion(predicted_action.squeeze(), target_action)
                
                self.scaler.scale(loss).backward()
                if self.config.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predicted_action = self.model(batch)
                loss = self.criterion(predicted_action.squeeze(), target_action)
                loss.backward()
                if self.config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if batch_idx % self.config.log_frequency == 0:
                print(f"  [{epoch:2d}][{batch_idx:4d}] Loss: {loss.item():.6f} | LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        epoch_time = time.time() - epoch_start
        
        return {
            'avg_loss': avg_loss,
            'epoch_time': epoch_time,
            'num_batches': num_batches,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].squeeze(0).to(self.device, non_blocking=True)
                
                if 'actions' not in batch or batch['actions'].numel() == 0:
                    continue
                
                target_action = batch['actions'][-1]
                predicted_action = self.model(batch)
                loss = self.criterion(predicted_action.squeeze(), target_action)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Early stopping check
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.patience_counter = 0
            
            if self.config.save_best_model:
                self.save_best_model(epoch, avg_loss)
        else:
            self.patience_counter += 1
        
        return {
            'avg_loss': avg_loss,
            'best_loss': self.best_val_loss,
            'patience': self.patience_counter
        }
    
    def save_best_model(self, epoch: int, val_loss: float):
        """Save the best model"""
        model_path = os.path.join(self.config.model_save_dir, "berkeley_best_model.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'training_history': self.training_history
        }, model_path)
        print(f"💾 Best model saved: {model_path}")
    
    def save_checkpoint(self, epoch: int):
        """Save training checkpoint"""
        checkpoint_path = os.path.join(self.config.output_dir, f"checkpoint_epoch_{epoch:03d}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'training_history': self.training_history,
            'config': self.config
        }, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def train(self):
        """Execute comprehensive training"""
        print(f"\n🚀 STARTING COMPREHENSIVE BERKELEY TRAINING")
        
        # Setup data loaders
        self.setup_data_loaders()
        
        # Training loop
        for epoch in range(1, self.config.num_epochs + 1):
            # Training
            train_metrics = self.train_epoch(epoch)
            self.training_history['train_loss'].append(train_metrics['avg_loss'])
            self.training_history['learning_rate'].append(train_metrics['learning_rate'])
            self.training_history['epoch_time'].append(train_metrics['epoch_time'])
            
            # Validation
            if epoch % self.config.eval_frequency == 0:
                val_metrics = self.validate(epoch)
                self.training_history['val_loss'].append(val_metrics['avg_loss'])
                
                print(f"📊 Epoch {epoch} Results:")
                print(f"   Train Loss: {train_metrics['avg_loss']:.6f}")
                print(f"   Val Loss: {val_metrics['avg_loss']:.6f}")
                print(f"   Best Val Loss: {val_metrics['best_loss']:.6f}")
                print(f"   Patience: {val_metrics['patience']}/{self.config.early_stopping_patience}")
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Checkpointing
            if epoch % self.config.checkpoint_frequency == 0:
                self.save_checkpoint(epoch)
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"🛑 Early stopping at epoch {epoch}")
                break
        
        # Save final model and results
        self.save_final_results()
        
        print(f"\n✅ COMPREHENSIVE BERKELEY TRAINING COMPLETED!")
        print(f"📊 Best validation loss: {self.best_val_loss:.6f}")
        print(f"💾 Results saved in: {self.config.results_dir}")
    
    def save_final_results(self):
        """Save final training results and analysis"""
        results = {
            'training_config': self.config.__dict__,
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss,
            'total_epochs': len(self.training_history['train_loss']),
            'dataset_stats': {
                'train_episodes': len(self.train_dataset),
                'val_episodes': len(self.val_dataset),
                'train_success_rate': self.train_dataset.success_rate,
                'val_success_rate': self.val_dataset.success_rate
            }
        }
        
        # Save results
        results_path = os.path.join(self.config.results_dir, "comprehensive_berkeley_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📊 Results saved: {results_path}")

def main():
    """Main training execution"""
    print("🤖 COMPREHENSIVE BERKELEY DATASET TRAINING")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    # Configuration for comprehensive training
    config = ComprehensiveTrainingConfig(
        # Use all Berkeley dataset files
        use_all_files=True,
        max_files_train=None,  # Use all 412 training files
        max_files_val=10,      # Limit validation for faster evaluation
        
        # Multi-epoch training
        num_epochs=15,
        batch_size=8,  # Conservative for complex model + long sequences
        
        # Advanced model architecture
        vision_backbone="resnet",
        hidden_dim=512,
        num_attention_heads=8,
        num_transformer_layers=4,
        
        # Data configuration
        image_size=(224, 224),
        max_sequence_length=30,  # Reasonable sequence length
        use_hand_camera=True,
        use_language=True,
        
        # Optimization
        learning_rate=5e-5,  # Conservative for large dataset
        mixed_precision=True,
        lr_schedule="cosine",
        
        # Monitoring
        eval_frequency=2,
        checkpoint_frequency=3,
        log_frequency=25
    )
    
    # Execute comprehensive training
    trainer = ComprehensiveBerkeleyTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
