################################################################################
# MIT License
#
# Copyright (c) 2025 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2025
# Date Created: 2025-10-28
################################################################################
"""
This module implements utility functions for plotting training results.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import matplotlib.pyplot as plt


def smooth_curve(values, window_size=50):
    """
    Apply moving average smoothing to a curve.
    
    Args:
        values: List or array of values to smooth
        window_size: Size of the smoothing window
    
    Returns:
        Smoothed values as numpy array
    """
    if len(values) < window_size:
        window_size = max(1, len(values) // 10)
    
    smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
    return smoothed


def save_training_plots(logging_dict, val_accuracies, test_accuracy, model_name='model', output_dir='plots', 
                        batches_per_epoch=None, smoothing_window=50):
    """
    Saves training plots in a single figure with three subplots.
    
    Args:
        logging_dict: Dictionary containing training metrics (loss, train_accuracy, val_accuracies)
        val_accuracies: List of validation accuracies per epoch
        test_accuracy: Final test accuracy
        model_name: Name of the model (e.g., 'numpy', 'pytorch') for file naming
        output_dir: Directory to save the plots
        batches_per_epoch: Number of batches per epoch (for vertical epoch lines). If None, auto-calculated.
        smoothing_window: Window size for moving average smoothing
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate batches per epoch if not provided
    if batches_per_epoch is None and len(logging_dict['loss']) > 0:
        num_epochs = len(val_accuracies)
        batches_per_epoch = len(logging_dict['loss']) // num_epochs if num_epochs > 0 else 1
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    fig.suptitle(f'{model_name.upper()} MLP Training Results (Test Acc: {test_accuracy:.4f})', 
                 fontsize=16, fontweight='bold')
    
    # ==================== Plot 1: Loss Curve ====================
    ax1 = axes[0]
    loss_values = np.array(logging_dict['loss'])
    batch_indices = np.arange(len(loss_values))
    
    # Plot raw loss (semi-transparent)
    ax1.plot(batch_indices, loss_values, alpha=0.3, color='tab:blue', linewidth=0.5, label='Raw Loss')
    
    # Plot smoothed loss
    if len(loss_values) > smoothing_window:
        smoothed_loss = smooth_curve(loss_values, smoothing_window)
        smooth_indices = batch_indices[:len(smoothed_loss)] + smoothing_window // 2
        ax1.plot(smooth_indices, smoothed_loss, color='tab:blue', linewidth=2, label=f'Smoothed Loss (window={smoothing_window})')
    
    # Add epoch separators
    if batches_per_epoch and batches_per_epoch > 0:
        for epoch in range(1, len(val_accuracies)):
            if epoch == 1:
                # Add label only to first line for legend
                ax1.axvline(x=epoch * batches_per_epoch, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Epoch Boundary')
            else:
                ax1.axvline(x=epoch * batches_per_epoch, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    ax1.set_xlabel('Batch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training Loss Curve', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # ==================== Plot 2: Train Accuracy Curve ====================
    ax2 = axes[1]
    train_acc_values = np.array(logging_dict['train_accuracy'])
    batch_indices = np.arange(len(train_acc_values))
    
    # Plot raw accuracy (semi-transparent)
    ax2.plot(batch_indices, train_acc_values, alpha=0.3, color='tab:orange', linewidth=0.5, label='Raw Accuracy')
    
    # Plot smoothed accuracy
    if len(train_acc_values) > smoothing_window:
        smoothed_acc = smooth_curve(train_acc_values, smoothing_window)
        smooth_indices = batch_indices[:len(smoothed_acc)] + smoothing_window // 2
        ax2.plot(smooth_indices, smoothed_acc, color='tab:orange', linewidth=2, label=f'Smoothed Accuracy (window={smoothing_window})')
    
    # Add epoch separators
    if batches_per_epoch and batches_per_epoch > 0:
        for epoch in range(1, len(val_accuracies)):
            if epoch == 1:
                # Add label only to first line for legend
                ax2.axvline(x=epoch * batches_per_epoch, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Epoch Boundary')
            else:
                ax2.axvline(x=epoch * batches_per_epoch, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    ax2.set_xlabel('Batch', fontsize=11)
    ax2.set_ylabel('Train Accuracy', fontsize=11)
    ax2.set_title('Training Accuracy Curve (per batch)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    
    # ==================== Plot 3: Training vs Validation Accuracy ====================
    ax3 = axes[2]
    epoch_indices = np.arange(1, len(val_accuracies) + 1)  # Start from 1 instead of 0
    
    # Plot validation accuracy
    ax3.plot(epoch_indices, val_accuracies, marker='o', markersize=6, 
             label='Validation Accuracy', linewidth=2, color='tab:green')
    
    # Mark best validation accuracy with a star
    best_val_idx = np.argmax(val_accuracies)
    best_val_acc = val_accuracies[best_val_idx]
    ax3.plot(best_val_idx + 1, best_val_acc, marker='*', markersize=20,  # +1 to match 1-indexed epochs
             color='gold', markeredgecolor='black', markeredgewidth=1.5,
             label=f'Best Val Acc: {best_val_acc:.4f} (Epoch {best_val_idx+1})', zorder=5)
    
    # Add training accuracy per epoch if available
    if 'train_accuracy_per_epoch' in logging_dict:
        ax3.plot(epoch_indices, logging_dict['train_accuracy_per_epoch'], 
                marker='s', markersize=6, label='Training Accuracy', linewidth=2, color='tab:orange')
    
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Accuracy', fontsize=11)
    ax3.set_title('Training vs Validation Accuracy (per epoch)', fontsize=12, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the combined figure
    output_path = os.path.join(output_dir, f'{model_name}_training_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training summary plot saved to '{output_path}'")

