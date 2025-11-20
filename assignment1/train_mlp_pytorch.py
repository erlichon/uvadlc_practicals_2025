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
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils
from plot_utils import save_training_plots
import json

import torch
import torch.nn as nn
import torch.optim as optim


def save_metrics_to_json(logging_dict, val_accuracies, test_accuracy, model_name='pytorch', plots_dir='plots'):
    """
    Saves training metrics to a JSON file.
    
    Args:
      logging_dict: Dictionary containing training metrics (loss, train_accuracy, etc.)
      val_accuracies: List of validation accuracies per epoch
      test_accuracy: Final test accuracy
      model_name: Name prefix for the JSON file
      plots_dir: Directory where to save the JSON file (same as plots directory)
    """
    # Ensure the directory exists
    os.makedirs(plots_dir, exist_ok=True)
    
    json_path = os.path.join(plots_dir, f'{model_name}_training_metrics.json')
    
    # Prepare metrics data
    metrics = {
        'epochs': list(range(1, len(val_accuracies) + 1)),
        'train_accuracy': logging_dict['train_accuracy_per_epoch'],
        'val_accuracy': val_accuracies,
        'test_accuracy': test_accuracy
    }
    
    # Write to JSON
    with open(json_path, 'w') as jsonfile:
        json.dump(metrics, jsonfile, indent=2)
    
    print(f"Metrics saved to {json_path}")


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.
    
    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
               each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    
    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    accuracy = (torch.sum(torch.argmax(predictions, dim=1) == targets) / targets.shape[0]).item()

    #######################
    # END OF YOUR CODE    #
    #######################
    
    return accuracy


def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset, 
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    model.eval()  # Set model to evaluation mode
    avg_accuracy = 0
    with torch.no_grad():  # Disable gradient computation for efficiency
      for batch in data_loader:
        x, y = batch
        x, y = x.to(model.device), y.to(model.device)
        predictions = model(x.view(x.shape[0], -1))
        avg_accuracy += accuracy(predictions, y)
    avg_accuracy /= len(data_loader)
    #######################
    # END OF YOUR CODE    #
    #######################
    
    return avg_accuracy


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation.
      logging_dict: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # TODO: Initialize model and loss module
    size = torch.prod(torch.tensor(cifar10['train'].dataset[0][0].shape)).item()
    model = MLP(n_inputs=size, n_hidden=hidden_dims, n_classes=10, use_batch_norm=use_batch_norm)
    model.to(device)  # Move model to device
    loss_module = nn.CrossEntropyLoss()
    # TODO: Test best model
    test_accuracy = 0
    best_model = None
    best_val_accuracy = 0
    # TODO: Add any information you might want to save for plotting
    logging_dict = {'loss': [], 'train_accuracy': [], 'train_accuracy_per_epoch': [], 'val_accuracies': []}
    val_accuracies = []
    # TODO: Do optimization with the simple SGD optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # TODO: Training loop including validation
    for epoch in range(epochs):
      model.train()
      epoch_train_acc = []
      for x, y in cifar10_loader['train']:
        x, y = x.to(device), y.to(device)
        x = x.view(x.shape[0], -1)
        optimizer.zero_grad()
        predictions = model(x)
        loss = loss_module(predictions, y)
        
        loss.backward()
        optimizer.step()
        batch_acc = accuracy(predictions, y)
        logging_dict['loss'].append(loss.item())
        logging_dict['train_accuracy'].append(batch_acc)
        epoch_train_acc.append(batch_acc)
      
      # log training accuracy for the epoch
      logging_dict['train_accuracy_per_epoch'].append(evaluate_model(model, cifar10_loader['train']))
      
      val_accuracy = evaluate_model(model, cifar10_loader['validation'])
      val_accuracies.append(val_accuracy)
      logging_dict['val_accuracies'].append(val_accuracy)
      if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model = deepcopy(model)
      
      print(f"Epoch {epoch+1}, Train Loss: {logging_dict['loss'][-1]}, Train Accuracy: {logging_dict['train_accuracy'][-1]}, Validation Accuracy: {val_accuracy}")
    
    # Evaluate best model on test set ONLY ONCE at the end
    test_accuracy = evaluate_model(best_model, cifar10_loader['test'])
    print(f"Final Test Accuracy (best validation model): {test_accuracy}")
    
    #######################
    # END OF YOUR CODE    #
    #######################

    return best_model, val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    model, val_accuracies, test_accuracy, logging_dict = train(**kwargs)
    
    # Determine model name and output directory
    if args.use_batch_norm:
      model_name = 'pytorch_batch_norm'
    else:
      model_name = 'pytorch'
    
    plots_dir = 'plots'
    
    # Save training plots to files
    save_training_plots(logging_dict, val_accuracies, test_accuracy, model_name=model_name, output_dir=plots_dir)
    
    # Save metrics to JSON in the same directory as plots
    save_metrics_to_json(logging_dict, val_accuracies, test_accuracy, model_name=model_name, plots_dir=plots_dir)