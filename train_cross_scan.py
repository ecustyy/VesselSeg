"""
Updated Training Script with Direction-Aware Innovations

This training script incorporates the novel direction-aware innovations:
1. Direction Consistency Loss - enforces directional continuity in predictions
2. Direction-Aware Attention - enhances feature extraction with directional sensitivity
3. Mixed CNN-Transformer Architecture - combines local and global feature learning

Author: Code Development Agent
Date: 2026-04-02
"""

import torch.backends.cudnn as cudnn
import torch.optim as optim
import sys, time
import os
from os.path import join
import torch
from torch.cuda.amp import autocast, GradScaler
import numpy as np

# Import updated config
from config_updated import parse_args, get_config_dict

# Import losses
from lib.losses.loss import *
from lib.losses.direction_loss import CombinedDirectionLoss, DirectionConsistencyLoss
from lib.losses.improved_direction_loss import CombinedDirectionLoss as ImprovedCombinedDirectionLoss

# Import utilities
from lib.common import *
from lib.logger import Logger, Print_Logger
from lib.metrics import Evaluate as compute_metrics

# Import models
import models
from test import Test

# Import data loading
from function import get_dataloader, train, val, get_dataloaderV2

# Import networks
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.direction_aware_attention import (
    DirectionAwareAttention, 
    DirectionAwareSpatialAttention,
    DirectionAwareTransformerBlock,
    create_direction_attention
)
from networks.direction_aware_hybrid import DirectionAwareHybridBottleneck
from networks.cross_scan_mamba import DirectionMambaEncoder, CrossScanBottleneckWrapper


def create_model(args, device):
    """
    Create model with direction-aware innovations.
    
    Args:
        args: Configuration arguments
        device: torch device
        
    Returns:
        nn.Module: Model with direction-aware components
    """
    # Create base Vision Transformer model
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.classes
    config_vit.n_skip = 3
    
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size), 
            int(args.img_size / args.vit_patches_size)
        )
    
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    
    # Add direction-aware attention if enabled
    if args.use_direction_attention:
        print(f"Adding direction-aware attention ({args.attention_type})...")
        
        # Depending on attention type, insert appropriate modules
        if args.attention_type == 'channel':
            attention_module = DirectionAwareAttention(
                in_channels=768,  # ViT-B hidden size
                reduction=args.attention_reduction,
                num_directions=args.num_directions
            )
        elif args.attention_type == 'spatial':
            attention_module = DirectionAwareSpatialAttention(
                in_channels=768
            )
        elif args.attention_type == 'transformer':
            # For transformer attention, we modify the existing attention blocks
            attention_module = DirectionAwareTransformerBlock(
                hidden_size=768,
                num_heads=12,
                mlp_dim=3072
            )
        
        # Attach attention module to network
        # Note: This is a simplified integration - for full integration,
        # you would modify the ViT architecture to include direction-aware attention
        # at strategic points in the encoder-decoder structure
        net.direction_attention = attention_module.to(device)
        net.use_direction_attention = True
    else:
        net.use_direction_attention = False
    
    # Add Hybrid Mamba-Transformer bottleneck if enabled
    if args.use_hybrid:
        print(f"Adding Cross-Scan Mamba Bottleneck (d_state={args.hybrid_d_state})...")
        
        cross_scan_mamba = DirectionMambaEncoder(
            dim=args.hybrid_d_model,
            num_layers=args.hybrid_num_layers,
            d_state=args.hybrid_d_state,
        )
        
        net = CrossScanBottleneckWrapper(net, cross_scan_mamba).to(device)
        print("Wrapped ViT model with Cross-Scan Mamba bottleneck")
    
    print(f"Model created with direction-aware innovations")
    print(f"Total number of parameters: {count_parameters(net):,}")
    
    return net


def create_criterion(args):
    """
    Create loss function with direction consistency.
    
    Args:
        args: Configuration arguments
        
    Returns:
        nn.Module or callable: Loss function
    """
    if args.use_direction_loss:
        print("Using Improved Combined Direction Loss (CE + Dynamic Direction Consistency)")
        print(f"  - Initial weight: {args.direction_loss_weight}")
        print(f"  - Max weight: {args.direction_loss_weight * 10}")
        print(f"  - Warmup epochs: 20")
        print(f"  - Boundary-focused: Enabled")
        criterion = ImprovedCombinedDirectionLoss(
            initial_weight=args.direction_loss_weight,
            max_weight=args.direction_loss_weight * 10,
            warmup_epochs=20,
            boundary_threshold=0.3,
            ce_weight=args.ce_weight
        )
    else:
        print("Using standard Cross-Entropy Loss")
        criterion = CrossEntropyLoss2d()
    
    return criterion


def train_epoch_with_direction_loss(train_loader, net, criterion, optimizer, 
                                     device, args, epoch):
    """
    Training epoch with direction-aware loss.
    
    Args:
        train_loader: DataLoader for training
        net: Network model
        criterion: Loss function
        optimizer: Optimizer
        device: torch device
        args: Configuration arguments
        epoch: Current epoch number
        
    Returns:
        dict: Training metrics
    """
    net.train()
    total_loss = 0
    total_ce_loss = 0
    total_dir_loss = 0
    num_batches = 0
    
    # Mixed precision scaler
    scaler = GradScaler() if args.use_mixed_precision else None
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if args.use_mixed_precision:
            with autocast():
                outputs = net(inputs)
                # Pass epoch only for direction loss (dynamic weight scheduling)
                if args.use_direction_loss:
                    loss_result = criterion(outputs, targets, epoch=epoch)
                else:
                    loss_result = criterion(outputs, targets)
                
                # Handle dict output from improved loss
                if isinstance(loss_result, dict):
                    loss = loss_result['total_loss']
                    ce_loss = loss_result['ce_loss'].item() if hasattr(loss_result['ce_loss'], 'item') else float(loss_result['ce_loss'])
                    dir_loss = loss_result['dir_loss'].item() if hasattr(loss_result['dir_loss'], 'item') else float(loss_result['dir_loss'])
                elif isinstance(loss_result, tuple):
                    loss, loss_dict = loss_result
                    ce_loss = loss_dict.get('cross_entropy', loss).item()
                    dir_loss = loss_dict.get('direction', 0).item()
                else:
                    loss = loss_result
                    ce_loss = loss.item()
                    dir_loss = 0.0
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if args.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.gradient_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = net(inputs)
            # Pass epoch only for direction loss (dynamic weight scheduling)
            if args.use_direction_loss:
                loss_result = criterion(outputs, targets, epoch=epoch)
            else:
                loss_result = criterion(outputs, targets)
            
            # Handle dict output from improved loss
            if isinstance(loss_result, dict):
                loss = loss_result['total_loss']
                ce_loss = loss_result['ce_loss'].item() if hasattr(loss_result['ce_loss'], 'item') else float(loss_result['ce_loss'])
                dir_loss = loss_result['dir_loss'].item() if hasattr(loss_result['dir_loss'], 'item') else float(loss_result['dir_loss'])
            elif isinstance(loss_result, tuple):
                loss, loss_dict = loss_result
                ce_loss = loss_dict.get('cross_entropy', loss).item()
                dir_loss = loss_dict.get('direction', 0).item()
            else:
                loss = loss_result
                ce_loss = loss.item()
                dir_loss = 0.0
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.gradient_clip)
            
            optimizer.step()
        
        # Accumulate metrics
        total_loss += loss
        total_ce_loss += ce_loss
        total_dir_loss += dir_loss
        
        num_batches += 1
        
        # Logging
        if (batch_idx + 1) % args.log_interval == 0:
            avg_loss = total_loss / num_batches
            avg_ce = total_ce_loss / num_batches if total_ce_loss > 0 else 0
            avg_dir = total_dir_loss / num_batches if total_dir_loss > 0 else 0
            
            print(f'Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] '
                  f'Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, Dir: {avg_dir:.4f})')
    
    # Compute epoch metrics
    metrics = {
        'train_loss': total_loss / num_batches,
        'train_ce_loss': total_ce_loss / num_batches if total_ce_loss > 0 else 0,
        'train_dir_loss': total_dir_loss / num_batches if total_dir_loss > 0 else 0
    }
    
    return metrics


def validate_with_direction_loss(val_loader, net, criterion, device, args):
    """
    Validation with direction-aware loss.
    
    Args:
        val_loader: DataLoader for validation
        net: Network model
        criterion: Loss function
        device: torch device
        args: Configuration arguments
        
    Returns:
        dict: Validation metrics
    """
    net.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = net(inputs)
            
            # Compute loss
            loss_result = criterion(outputs, targets)
            if isinstance(loss_result, dict):
                loss = loss_result['total_loss']
            elif isinstance(loss_result, tuple):
                loss, _ = loss_result
            else:
                loss = loss_result
            
            total_loss += loss.item()
            
            # Collect predictions and targets
            if outputs.shape[1] == 2:
                preds = torch.argmax(outputs, dim=1)
            else:
                preds = torch.sigmoid(outputs[:, 0]) > 0.5
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Compute metrics using Evaluate class
    evaluator = compute_metrics()
    evaluator.add_batch(all_targets, all_preds)
    metrics = {
        'val_auc_roc': evaluator.auc_roc(),
        'val_sensitivity': evaluator.sensitivity(),
        'val_specificity': evaluator.specificity()
    }
    metrics['val_loss'] = total_loss / len(val_loader)
    
    return metrics


def main():
    """Main training function with direction-aware innovations."""
    
    # Set random seed for reproducibility
    setpu_seed(2021)
    
    # Parse arguments
    args = parse_args()
    save_path = join(args.outf, args.save)
    save_args(args, save_path)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    cudnn.benchmark = True
    
    # Setup logging
    log = Logger(save_path)
    sys.stdout = Print_Logger(os.path.join(save_path, 'train_log.txt'))
    print('The computing device used is: ', 'GPU' if device.type == 'cuda' else 'CPU')
    print(f"Configuration: {get_config_dict(args)}")
    
    # Create model with direction-aware innovations
    net = create_model(args, device)
    
    # Load pre-trained checkpoint if specified
    if args.pre_trained is not None:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.outf + '%s/latest_model.pth' % args.pre_trained)
        net.load_state_dict(checkpoint['net'])
        args.start_epoch = checkpoint['epoch'] + 1
    
    # Create loss function
    criterion = create_criterion(args)
    
    # Create optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.N_epochs, eta_min=0
    )
    
    # Create data loaders
    train_loader, val_loader = get_dataloaderV2(args)
    
    # Setup validation on test set if specified
    if args.val_on_test: 
        print('\033[0;32m===============Validation on Testset!!!===============\033[0m')
        val_tool = Test(args)
    
    # Training tracking - Save top 5 models
    best_models = []  # List of (epoch, auc_roc) tuples
    top_k = 5  # Save top 5 models
    trigger = 0  # Early stopping counter (disabled for full 50 epochs)
    
    print("\n" + "="*60)
    print("Starting Training with Direction-Aware Innovations")
    print("="*60)
    print(f"Direction Consistency Loss: {args.use_direction_loss}")
    print(f"Direction-Aware Attention: {args.use_direction_attention}")
    print(f"Attention Type: {args.attention_type}")
    print(f"Mixed Precision: {args.use_mixed_precision}")
    print(f"Save Top-{top_k} Models: Enabled")
    print(f"Early Stopping: Disabled (full 50 epochs)")
    print("="*60 + "\n")
    
    # Training loop - always run full 50 epochs
    for epoch in range(args.start_epoch, args.N_epochs + 1):
        print('\nEPOCH: %d/%d --(learn_rate:%.6f) | Time: %s' % \
            (epoch, args.N_epochs, optimizer.state_dict()['param_groups'][0]['lr'], 
             time.asctime()))
        
        # Training stage
        train_log = train_epoch_with_direction_loss(
            train_loader, net, criterion, optimizer, device, args, epoch
        )
        
        # Validation stage
        if not args.val_on_test:
            val_log = validate_with_direction_loss(val_loader, net, criterion, device, args)
        else:
            val_tool.inference(net)
            val_log = val_tool.val()
        
        # Update logger
        log.update(epoch, train_log, val_log)
        lr_scheduler.step()
        
        # Save checkpoint
        state = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'val_auc_roc': val_log['val_auc_roc'],
            'val_sensitivity': val_log.get('val_sensitivity', 0),
            'val_specificity': val_log.get('val_specificity', 0),
            'val_loss': val_log['val_loss']
        }
        
        # Save latest model (every 5 epochs)
        if epoch % 5 == 0:
            torch.save(state, join(save_path, f'epoch_{epoch:03d}_model.pth'))
        
        # Save top-k best models
        current_auc = val_log['val_auc_roc']
        best_models.append((epoch, current_auc))
        best_models.sort(key=lambda x: x[1], reverse=True)  # Sort by AUC descending
        
        # Keep only top-k
        if len(best_models) > top_k:
            # Remove the worst model from the list (but don't delete file)
            best_models = best_models[:top_k]
        
        # Save all top-k models
        for i, (ep, auc) in enumerate(best_models):
            if ep == epoch:  # This is the current epoch model
                torch.save(state, join(save_path, f'top{i+1}_model.pth'))
                print(f'\033[0;33mSaving top-{i+1} model (Epoch {ep}, AUC: {auc:.4f})!\033[0m')
        
        # Print best performance
        print('Top-5 Best Models:')
        for i, (ep, auc) in enumerate(best_models):
            print(f'  Rank {i+1}: Epoch {ep} | AUC_roc: {auc:.6f}')
        
        # Early stopping - DISABLED to ensure full 50 epochs
        # if args.early_stop is not None:
        #     if trigger >= args.early_stop:
        #         print("=> Early stopping")
        #         break
        
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    # Save final summary
    print("\n" + "="*60)
    print("Training Completed! (Full 50 epochs)")
    print(f"Top-5 Best Models saved to: {save_path}")
    print("\nFinal Top-5 Ranking:")
    for i, (ep, auc) in enumerate(best_models):
        print(f'  Rank {i+1}: Epoch {ep} | AUC-ROC: {auc:.6f}')
    print("="*60)
    
    # Save best model info to file
    with open(join(save_path, 'top5_models.txt'), 'w') as f:
        f.write("Top-5 Best Models\n")
        f.write("="*50 + "\n")
        for i, (ep, auc) in enumerate(best_models):
            f.write(f"Rank {i+1}: Epoch {ep} | AUC-ROC: {auc:.6f}\n")


if __name__ == '__main__':
    main()
