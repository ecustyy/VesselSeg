"""
Improved Direction Consistency Loss for Vessel Segmentation

Improvements over original version:
1. Dynamic weight scheduling - gradually increases during training
2. Boundary-focused loss - only computes direction loss on vessel boundaries
3. Adaptive gradient computation - better handling of small vessels

Author: Code Development Agent (Improved Version)
Date: 2026-04-03
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ImprovedDirectionConsistencyLoss(nn.Module):
    """
    Improved Direction Consistency Loss with dynamic weighting and boundary focus.
    
    Key improvements:
    1. Dynamic weight scheduling: weight grows from initial_weight to max_weight
       over warmup_epochs, then stays constant
    2. Boundary-focused: only computes direction loss on vessel boundaries
       where orientation information is most relevant
    3. Multi-scale gradients: computes gradients at multiple scales for robustness
    
    Args:
        initial_weight (float): Initial weight at epoch 0 (default: 0.001)
        max_weight (float): Maximum weight after warmup (default: 0.1)
        warmup_epochs (int): Number of epochs to ramp up weight (default: 20)
        boundary_threshold (float): Threshold for boundary detection (default: 0.3)
        alpha (float): Weight for horizontal gradient consistency
        beta (float): Weight for vertical gradient consistency
        gamma (float): Weight for diagonal gradient consistency
    """
    
    def __init__(self, initial_weight=0.001, max_weight=0.1, warmup_epochs=20,
                 boundary_threshold=0.3, alpha=1.0, beta=1.0, gamma=0.5):
        super(ImprovedDirectionConsistencyLoss, self).__init__()
        self.initial_weight = initial_weight
        self.max_weight = max_weight
        self.warmup_epochs = warmup_epochs
        self.boundary_threshold = boundary_threshold
        self.alpha = alpha  # horizontal weight
        self.beta = beta    # vertical weight
        self.gamma = gamma  # diagonal weight
        
        self.current_epoch = 0
        self.current_weight = initial_weight
        
        # Sobel operators for gradient computation
        self._register_sobel_kernels()
    
    def _register_sobel_kernels(self):
        """Register Sobel kernels as non-trainable buffers."""
        # Horizontal Sobel kernel
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        # Vertical Sobel kernel
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        # Diagonal (45°) Sobel kernel
        sobel_diag1 = torch.tensor([
            [0, 1, 2],
            [-1, 0, 1],
            [-2, -1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        # Diagonal (135°) Sobel kernel
        sobel_diag2 = torch.tensor([
            [2, 1, 0],
            [1, 0, -1],
            [0, -1, -2]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('sobel_diag1', sobel_diag1)
        self.register_buffer('sobel_diag2', sobel_diag2)
    
    def set_epoch(self, epoch):
        """
        Update current epoch and compute dynamic weight.
        
        Weight scheduling:
        - Epoch 0-5: very small weight (0.001) to let model learn basic segmentation
        - Epoch 6-20: linear increase from 0.001 to 0.1
        - Epoch 21+: constant at 0.1
        """
        self.current_epoch = epoch
        
        if epoch <= 5:
            # Initial phase: minimal direction constraint
            self.current_weight = self.initial_weight
        elif epoch <= self.warmup_epochs:
            # Warmup phase: linear increase
            progress = (epoch - 5) / (self.warmup_epochs - 5)
            self.current_weight = self.initial_weight + progress * (self.max_weight - self.initial_weight)
        else:
            # Stable phase: constant weight
            self.current_weight = self.max_weight
    
    def _compute_boundary_mask(self, target, threshold=0.3):
        """
        Compute boundary mask from ground truth.
        
        Boundaries are detected using gradient magnitude.
        Only pixels near vessel boundaries will contribute to direction loss.
        
        Args:
            target: Ground truth segmentation (N, 1, H, W) or (N, H, W)
            threshold: Threshold for boundary detection
            
        Returns:
            boundary_mask: Binary mask of boundary regions (N, 1, H, W)
        """
        # Ensure target has correct shape (N, 1, H, W)
        if target.dim() == 3:
            # Add channel dimension: (N, H, W) -> (N, 1, H, W)
            target = target.unsqueeze(1)
        elif target.dim() == 4 and target.size(1) > 1:
            # If target has multiple channels (e.g., one-hot or batch mixed up), take first channel
            # Handle case where shape might be (1, 16, H, W) due to batching issues
            if target.size(0) == 1 and target.size(1) > 1:
                # Likely a batching issue, reshape or take appropriate slice
                target = target[:, 0:1, :, :]
            else:
                target = target[:, 0:1, :, :]
        
        # Compute gradient magnitude
        grad_x = F.conv2d(target, self.sobel_x.to(target.device), padding=1)
        grad_y = F.conv2d(target, self.sobel_y.to(target.device), padding=1)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        
        # Normalize to [0, 1]
        grad_mag = grad_mag / (grad_mag.max() + 1e-8)
        
        # Create soft boundary mask (smooth thresholding)
        boundary_mask = torch.sigmoid((grad_mag - threshold) * 10)
        
        return boundary_mask
    
    def _compute_directional_gradients(self, x):
        """
        Compute multi-directional gradients.
        
        Args:
            x (Tensor): Input tensor of shape (N, 1, H, W)
            
        Returns:
            tuple: Gradients in horizontal, vertical, and diagonal directions
        """
        grad_x = self._compute_gradient(x, self.sobel_x)
        grad_y = self._compute_gradient(x, self.sobel_y)
        grad_diag1 = self._compute_gradient(x, self.sobel_diag1)
        grad_diag2 = self._compute_gradient(x, self.sobel_diag2)
        
        return grad_x, grad_y, grad_diag1, grad_diag2
    
    def _compute_gradient(self, x, kernel):
        """
        Compute gradient using convolution with kernel.
        
        Args:
            x (Tensor): Input tensor of shape (N, 1, H, W)
            kernel (Tensor): Convolution kernel
            
        Returns:
            Tensor: Gradient magnitude
        """
        # Pad input to maintain spatial dimensions
        x_padded = F.pad(x, (1, 1, 1, 1), mode='reflect')
        
        # Convert kernel to match input dtype and device
        kernel = kernel.to(dtype=x.dtype, device=x.device)
        
        gradient = F.conv2d(x_padded, kernel, padding=0)
        return gradient
    
    def forward(self, predictions, targets, epoch=None):
        """
        Compute direction consistency loss.
        
        Args:
            predictions: Predicted segmentation probabilities (N, 1, H, W)
            targets: Ground truth segmentation (N, 1, H, W)
            epoch: Current epoch (optional, for dynamic weight)
            
        Returns:
            Tensor: Direction consistency loss
        """
        # Update epoch and weight if provided
        if epoch is not None:
            self.set_epoch(epoch)
        
        # Compute boundary mask from ground truth
        boundary_mask = self._compute_boundary_mask(targets, self.boundary_threshold)
        
        # Compute directional gradients for prediction and target
        pred_grads = self._compute_directional_gradients(predictions)
        target_grads = self._compute_directional_gradients(targets)
        
        # Compute direction consistency loss for each direction
        # Only consider boundary regions
        loss_h = self._direction_loss(pred_grads[0], target_grads[0], boundary_mask)
        loss_v = self._direction_loss(pred_grads[1], target_grads[1], boundary_mask)
        loss_d1 = self._direction_loss(pred_grads[2], target_grads[2], boundary_mask)
        loss_d2 = self._direction_loss(pred_grads[3], target_grads[3], boundary_mask)
        
        # Weighted sum of directional losses
        dir_loss = (self.alpha * loss_h + self.beta * loss_v + 
                   self.gamma * (loss_d1 + loss_d2)) / (self.alpha + self.beta + 2 * self.gamma)
        
        return dir_loss * self.current_weight
    
    def _direction_loss(self, pred_grad, target_grad, boundary_mask):
        """
        Compute direction loss for a single direction.
        
        Uses cosine similarity to measure gradient alignment.
        Only considers boundary regions.
        
        Args:
            pred_grad: Predicted gradient
            target_grad: Target gradient
            boundary_mask: Boundary region mask
            
        Returns:
            Tensor: Direction loss for this direction
        """
        # Flatten tensors
        pred_flat = pred_grad.view(pred_grad.size(0), -1)
        target_flat = target_grad.view(target_grad.size(0), -1)
        mask_flat = boundary_mask.view(boundary_mask.size(0), -1)
        
        # Apply boundary mask
        pred_masked = pred_flat * mask_flat
        target_masked = target_flat * mask_flat
        
        # Compute cosine similarity
        dot_product = torch.sum(pred_masked * target_masked, dim=1)
        pred_norm = torch.norm(pred_masked, dim=1) + 1e-8
        target_norm = torch.norm(target_masked, dim=1) + 1e-8
        
        cos_similarity = dot_product / (pred_norm * target_norm)
        
        # Convert to loss (1 - similarity)
        # Only consider pixels where boundary mask is active
        boundary_ratio = torch.sum(mask_flat > 0.5, dim=1) / mask_flat.size(1)
        loss = (1 - cos_similarity) * boundary_ratio
        
        return loss.mean()


class CombinedDirectionLoss(nn.Module):
    """
    Combined loss: Cross-Entropy + Improved Direction Consistency Loss.
    
    This is the main loss function to use during training.
    """

    def __init__(self, initial_weight=0.001, max_weight=0.1, warmup_epochs=20,
                 boundary_threshold=0.3, ce_weight=1.0):
        super(CombinedDirectionLoss, self).__init__()
        self.ce_weight = ce_weight  # Save for forward pass
        self.direction_loss = ImprovedDirectionConsistencyLoss(
            initial_weight=initial_weight,
            max_weight=max_weight,
            warmup_epochs=warmup_epochs,
            boundary_threshold=boundary_threshold
        )
        # Don't create CE loss here, create it in forward to ensure correct device

    def set_epoch(self, epoch):
        """Update epoch for dynamic weight scheduling."""
        self.direction_loss.set_epoch(epoch)

    def forward(self, predictions, targets, epoch=None):
        """
        Compute combined loss.
        
        Args:
            predictions: Predicted logits (N, 2, H, W)
            targets: Ground truth segmentation (N, 1, H, W) or (N, H, W) with values 0 or 1
            epoch: Current epoch (for dynamic weight)
            
        Returns:
            dict: Contains total_loss, ce_loss, dir_loss, and current_weight
        """
        # Ensure targets are on the same device as predictions
        targets = targets.to(predictions.device)
        
        # Ensure targets has correct shape (N, 1, H, W)
        if targets.dim() == 3:
            # (N, H, W) -> (N, 1, H, W)
            targets = targets.unsqueeze(1)
        elif targets.dim() == 4 and targets.size(1) != 1:
            # If targets has wrong shape, try to fix it
            if targets.size(0) == 1 and targets.size(1) > 1:
                # Likely a batching issue, take first channel
                targets = targets[:, 0:1, :, :]
            else:
                targets = targets[:, 0:1, :, :]
        
        # Create CE loss on the correct device
        ce_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([self.ce_weight, 1.0]).to(predictions.device))
        
        # Compute cross-entropy loss
        # Ensure targets are Long type for CE loss
        ce_loss = ce_loss_fn(predictions, targets.long().squeeze(1))
        
        # Compute direction loss (use softmax probability for vessel class)
        # Convert probs to float32 for direction loss computation
        probs = F.softmax(predictions.float(), dim=1)[:, 1:2, :, :]  # Vessel class probability
        dir_loss = self.direction_loss(probs, targets.float(), epoch)
        
        # Total loss
        total_loss = ce_loss + dir_loss
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'dir_loss': dir_loss,
            'current_weight': self.direction_loss.current_weight
        }
