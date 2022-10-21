"""
This script contains the FFN class used to impute the missing values
"""


import torch.nn as nn
import torch.nn.functional as F


class FFNImputer(nn.Module):

    def __init__(self, depth = 1, width = 32, dropout = 0.0, batch_normalize = False):
        super().__init__()

        # create ModuleDict for the nn structure
        self.layers = nn.ModuleDict()
        self.depth  = depth

        # define input layer
        self.layers["input"] = nn.Linear(10, width)

        # define hidden layers
        for layer in range(self.depth):

            self.layers[f"batchnorm{layer}"] = nn.BatchNorm1d(width)
            self.layers[f"hidden{layer}"]    = nn.Linear(width, width)
        
        # define output layer
        self.layers["output"] = nn.Linear(width, 1)

        # define dropout
        self.dropout_toggle = dropout > 0.0
        self.dropout        = nn.Dropout(dropout)

        # define batch_normalize toggle
        self.batch_normalize_toggle = batch_normalize


    def forward(self, x):

        # pass through input layer + activation
        x = F.relu( self.layers["input"](x) )

        # pass through hidden layers + activation functions
        for layer in range(self.depth):
            
            if self.batch_normalize_toggle:
                x = self.layers[f"batchnorm{layer}"](x)

            x = F.relu( self.layers[f"hidden{layer}"](x) )

            if self.dropout_toggle:
                x = self.dropout(x)
            
        # pass through output layer and return
        return self.layers["output"](x)
        