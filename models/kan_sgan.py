import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bsrbf_kan import BSRBF_KANLayer
from .fast_kan import FastKANLayer
from .faster_kan import FasterKANLayer
from .efficient_kan import EfficientKANLinear

class KAN_Generator(torch.nn.Module):
    
    def __init__(
        self, 
        layers_hidden,
        grid_size=5,
        spline_order=3,  
        base_activation=torch.nn.SiLU, 

        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8, # original: 8
        use_base_update: bool = True,
        spline_weight_init_scale: float = 0.1,
        
        exponent: int = 2,
        inv_denominator: float = 0.5,
        train_grid: bool = False,     
        train_inv_denominator: bool = False,
         
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        grid_eps=0.02,
        grid_range=[-1, 1],
        
        kan_layer = 'bsrbf_kan',
    ):
        super(KAN_Generator, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        #self.drop = torch.nn.Dropout(p=0.1) # dropout
        #self.tanh = nn.Tanh()
        
        for input_dim, output_dim in zip(layers_hidden, layers_hidden[1:]):
            
            if (kan_layer == 'bsrbf_kan'):
                self.layers.append(
                    BSRBF_KANLayer(
                        input_dim,
                        output_dim,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        base_activation=base_activation,
                        )
                )
            elif(kan_layer == 'fast_kan'):
                self.layers.append(
                    FastKANLayer(
                        input_dim, 
                        output_dim,
                        grid_min=grid_min,
                        grid_max=grid_max,
                        num_grids=num_grids,
                        use_base_update=use_base_update,
                        base_activation=base_activation,
                        spline_weight_init_scale=spline_weight_init_scale,
                        )
                )
            elif(kan_layer == 'faster_kan'):
                self.layers.append(
                    FasterKANLayer(
                        input_dim, 
                        output_dim,
                        grid_min=grid_min,
                        grid_max=grid_max,
                        num_grids=num_grids,
                        exponent = exponent,
                        inv_denominator = inv_denominator,
                        train_grid = train_grid,
                        train_inv_denominator = train_inv_denominator,
                        #use_base_update=use_base_update,
                        base_activation=base_activation,
                        spline_weight_init_scale=spline_weight_init_scale,
                        )
                )
            elif (kan_layer == 'efficient_kan'):
                self.layers.append(
                    EfficientKANLinear(
                        input_dim,
                        output_dim,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )   
                )

    def forward(self, x: torch.Tensor):
        #x = self.drop(x)
        
        for layer in self.layers: 
            x = layer(x)
        #x = self.tanh(x)
        return x

class KAN_Discriminator(torch.nn.Module):
    """
        Real data vs. fake data + Multiclass classification
    """
    
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,  
        base_activation=torch.nn.SiLU, 

        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8, # original: 8
        use_base_update: bool = True,
        spline_weight_init_scale: float = 0.1,
        
        exponent: int = 2,
        inv_denominator: float = 0.5,
        train_grid: bool = False,     
        train_inv_denominator: bool = False,
         
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        grid_eps=0.02,
        grid_range=[-1, 1],
        
        kan_layer = 'bsrbf_kan',
    ):
        super(KAN_Discriminator, self).__init__()
        self.layers = torch.nn.ModuleList()
        #self.drop = torch.nn.Dropout(p=0.05) # dropout
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        self.fc_adv = nn.Linear(layers_hidden[-1], 1)  # Adversarial output
        #self.fc_aux = nn.Linear(layers_hidden[-2], layers_hidden[-1])  # Auxiliary output
        self.sigmoid = nn.Sigmoid()
        #self.softmax = nn.Softmax(dim=1)
        
        #self.adv_weight = torch.nn.Parameter(torch.Tensor(1, layers_hidden[-2]))
        #torch.nn.init.kaiming_uniform_(self.adv_weight, a=math.sqrt(5))

        for input_dim, output_dim in zip(layers_hidden, layers_hidden[1:]):
            if (kan_layer == 'bsrbf_kan'):
                self.layers.append(
                    BSRBF_KANLayer(
                        input_dim,
                        output_dim,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        base_activation=base_activation,
                        )
                )
            elif(kan_layer == 'fast_kan'):
                self.layers.append(
                    FastKANLayer(
                        input_dim, 
                        output_dim,
                        grid_min=grid_min,
                        grid_max=grid_max,
                        num_grids=num_grids,
                        use_base_update=use_base_update,
                        base_activation=base_activation,
                        spline_weight_init_scale=spline_weight_init_scale,
                        )
                )
            elif(kan_layer == 'faster_kan'):
                self.layers.append(
                    FasterKANLayer(
                        input_dim, 
                        output_dim,
                        grid_min=grid_min,
                        grid_max=grid_max,
                        num_grids=num_grids,
                        exponent = exponent,
                        inv_denominator = inv_denominator,
                        train_grid = train_grid,
                        train_inv_denominator = train_inv_denominator,
                        #use_base_update=use_base_update,
                        base_activation=base_activation,
                        spline_weight_init_scale=spline_weight_init_scale,
                        )
                )
            elif (kan_layer == 'efficient_kan'):
                self.layers.append(
                    EfficientKANLinear(
                        input_dim,
                        output_dim,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )   
                )
    
    def forward(self, x: torch.Tensor):
       
        for layer in self.layers: 
            x = layer(x)
        #x = self.drop(x)
        
        adv_out = self.sigmoid(self.fc_adv(x))  # Adversarial output
        #adv_out = self.fc_adv(x)  # Adversarial output
        #aux_out = self.softmax(self.fc_aux(x))  # Auxiliary output

        return adv_out, x
