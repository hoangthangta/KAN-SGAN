from .efficient_kan import EfficientKANLinear, EfficientKAN
from .fast_kan import FastKANLayer, FastKAN, AttentionWithFastKANTransform
from .faster_kan import FasterKAN
from .bsrbf_kan import BSRBF_KAN
from .mlp import MLP

from .mlp_gan import MLP_Discriminator, MLP_Generator
from .kan_sgan import KAN_Generator, KAN_Discriminator

__all__ = ["EfficientKAN", "EfficientKANLinear", "FastKAN", "FasterKAN", "BSRBF_KAN", "MLP", "MLP_Discriminator", "MLP_Generator", "KAN_Generator", "KAN_Discriminator"]