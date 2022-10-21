"""
This script contains a function to create an instance of our FFN, the
optimizer, and the loss function
"""

import torch
import torch.nn as nn
from FFNImputer import FFNImputer

def initiate_model(depth = 1,
                   width = 32,
                   dropout = 0.0,
                   batch_normalize = False,
                   learning_rate = 0.001):

    # Create model instance, loss function, and optimizer
    # Model instance
    model_instance = FFNImputer(depth = depth,
                                      width = width,
                                      dropout = dropout,
                                      batch_normalize = batch_normalize)

    ## Loss function
    loss_function = nn.MSELoss()

    ## Optimizer
    optimizer = torch.optim.Adam(model_instance.parameters(),
                                 lr = learning_rate)

    return model_instance, loss_function, optimizer
