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
import matplotlib.pyplot as plt


def save_training_plots(logging_dict, val_accuracies, test_accuracy, model_name='model', output_dir='plots'):
    """
    Saves training plots to files with indicative names.
    
    Args:
        logging_dict: Dictionary containing training metrics (loss, train_accuracy, val_accuracies)
        val_accuracies: List of validation accuracies per epoch
        test_accuracy: Final test accuracy
        model_name: Name of the model (e.g., 'numpy', 'pytorch') for file naming
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot and save loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(logging_dict['loss'])
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve - {model_name.upper()} MLP Training')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{model_name}_train_loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot and save train accuracy curve
    plt.figure(figsize=(10, 6))
    plt.plot(logging_dict['train_accuracy'])
    plt.xlabel('Batch')
    plt.ylabel('Train Accuracy')
    plt.title(f'Train Accuracy Curve - {model_name.upper()} MLP Training')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{model_name}_train_accuracy_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot and save training vs validation accuracy curve
    plt.figure(figsize=(10, 6))
    plt.plot(val_accuracies, label='Validation Accuracy', linewidth=2)
    
    # Add training accuracy per epoch if available
    if 'train_accuracy_per_epoch' in logging_dict:
        plt.plot(logging_dict['train_accuracy_per_epoch'], label='Training Accuracy', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Training vs Validation Accuracy - {model_name.upper()} MLP (Best Test Acc: {test_accuracy:.4f})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{model_name}_train_validation_accuracy_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training plots saved to '{output_dir}/' directory with prefix '{model_name}_'")

