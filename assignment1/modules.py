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
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class ModuleList(object):
  def __init__(self, modules):
    self.modules = modules
  
  def forward(self, x):
    for module in self.modules:
      x = module.forward(x)
    return x
  
  def backward(self, dout):
    for module in reversed(self.modules):
      dout = module.backward(dout)
    return dout
  
  def clear_cache(self):
    for module in self.modules:
      module.clear_cache()
      
  def parameters(self):
    """
    Returns a list of parameter dictionaries from all modules that have them.
    """
    params = []
    for module in self.modules:
      if hasattr(module, 'params'):
        params.append(module.params)
    return params
  
  def gradients(self):
    """
    Returns a list of gradient dictionaries from all modules that have them.
    """
    grads = []
    for module in self.modules:
      if hasattr(module, 'grads'):
        grads.append(module.grads)
    return grads
  


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization. 
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """

        # Note: For the sake of this assignment, please store the parameters
        # and gradients in this format, otherwise some unit tests might fail.
        self.params = {'weight': None, 'bias': None} # Model parameters
        self.grads = {'weight': None, 'bias': None} # Gradients

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # ELU has similar variance to ReLU, so we use the same variance for the weight initialization.
        if input_layer:
          # first layer doesn't have activation function, so we don't need to multiply by sqrt(2). We assume 
          self.params['weight'] = np.random.randn(in_features, out_features) * np.sqrt(1 / out_features)
        else:
          self.params['weight'] = np.random.randn(out_features, in_features) * np.sqrt(2 / in_features) / np.sqrt(out_features)
  
        self.params['bias'] = np.zeros(out_features)
        self.grads['weight'] = np.zeros((in_features, out_features))
        self.grads['bias'] = np.zeros(out_features)
        
        self.cache = {'x': None}
  
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        out = x @ self.params['weight'].T + self.params['bias']
        self.cache['x'] = x
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # We use the equations developed in question 1 of HW1.
        if isinstance(self.cache['x'], np.ndarray):
          self.grads['weight'] = dout.T @ self.cache['x']
          self.grads['bias'] = np.sum(dout, axis=0) # dL/db = dL/dy^T @ dy/db = dL/dy^T @ 1 
          dx = dout @ self.params['weight']
        else:
          raise ValueError("Cache is not set. Probably forward pass was not called prior to backward pass.")
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        self.cache['x'] = None
        #######################
        # END OF YOUR CODE    #
        #######################


class ELUModule(object):
    """
    ELU activation module.
    """

    def __init__(self, alpha):
        self.alpha = alpha
        self.cache = {'x': None}

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # according to the formula in the lecture notes
        # we use the where function to implement the ELU function in vectorized form.
        out = np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
        self.cache['x'] = x
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # according to the formula in the lecture notes
        # we use the where function to implement the ELU function in vectorized form.
        if isinstance(self.cache['x'], np.ndarray):
          dx = np.where(self.cache['x'] > 0, dout, self.alpha * np.exp(self.cache['x']) * dout)
        else:
          raise ValueError("Cache is not set. Probably forward pass was not called prior to backward pass.")
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.cache['x'] = None
        #######################
        # END OF YOUR CODE    #
        #######################


class SoftMaxModule(object):
    """
    Softmax activation module.
    """
    
    def __init__(self):
        self.cache = {'x': None}

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        m = np.max(x, axis=1, keepdims=True)
        # using the max trick to stabilize the computation
        out = np.exp(x - m) / np.sum(np.exp(x - m), axis=1, keepdims=True)
        self.cache['out'] = out
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # according to the formula in the lecture notes
        if isinstance(self.cache['out'], np.ndarray):
          dx = np.sum(dout[..., :, None] * (self.cache['out'][..., :, None] * (np.eye(self.cache['out'].shape[-1])[None, ...] - self.cache['out'][..., None, :])), axis=1)
        else:
          raise ValueError("Cache is not set. Probably forward pass was not called prior to backward pass.")
        
        #######################
        # END OF YOUR CODE    #
        #######################

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.cache['out'] = None
        #######################
        # END OF YOUR CODE    #
        #######################


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # We use the cross entropy loss formula: L = -1/N * sum(y_i * log(p_i)). 
        # X is the output of the softmax layer, which is the probability of each class.
        out = -np.sum(np.log(x[np.arange(len(y)), y])) / x.shape[0]
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """
        

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # according to the formula in the lecture notes
        dx = np.zeros_like(x)  # Start with all zeros, shape (N, C)

        dx[np.arange(len(y)), y] = -1 / (x[np.arange(len(y)), y] * x.shape[0])
        #######################
        # END OF YOUR CODE    #
        #######################

        return dx