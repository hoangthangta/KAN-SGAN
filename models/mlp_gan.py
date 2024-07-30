import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLPLayer

      
class MLP_Generator(torch.nn.Module):
    
    def __init__(
        self, 
        layers_hidden,
        base_activation=torch.nn.SiLU,
    ):
        super(MLP_Generator, self).__init__()
        self.layers = torch.nn.ModuleList()
        #self.tanh = torch.nn.Tanh()
        #self.layernorm = nn.LayerNorm(layers_hidden[0])
        
        for input_dim, output_dim in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                MLPLayer(
                    input_dim,
                    output_dim,
                    base_activation=base_activation,
                )
            )
    
    def forward(self, x: torch.Tensor):
        #x = self.layernorm(x)
        for layer in self.layers: 
            x = layer(x)  
        #x = self.tanh(x)
        return x


class MLP_Discriminator(torch.nn.Module):
    
    def __init__(
        self, 
        layers_hidden,
        base_activation=torch.nn.SiLU,
    ):
        super(MLP_Discriminator, self).__init__()
        self.layernorm = nn.LayerNorm(layers_hidden[0])
        self.layers = torch.nn.ModuleList()
        self.fc_adv = nn.Linear(layers_hidden[-1], 1)  # Adversarial output
        #self.fc_aux = nn.Linear(layers_hidden[-2], layers_hidden[-1])  # Auxiliary output
        self.sigmoid = nn.Sigmoid()
        #self.softmax = nn.Softmax(dim=1)
        #self.dropout = nn.Dropout(p=0.1)
        
        for input_dim, output_dim in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                MLPLayer(
                    input_dim,
                    output_dim,
                    base_activation=base_activation,
                )
            )
    
    def forward(self, x: torch.Tensor):
        #x = self.layernorm(x)
        #x = self.dropout(x)
        for layer in self.layers: 
            x = layer(x)

        adv_out = self.sigmoid(self.fc_adv(x))  # Adversarial output
        #aux_out = self.softmax(self.fc_aux(x)) # Auxiliary output
        
        return adv_out, x