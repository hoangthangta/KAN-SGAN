# KAN-SGAN
Semi-supervised learning with Generative Adversarial Networks (SGANs) using Kolmogorov-Arnold Network Layers (KANLs)

In this repo, we set up GANs using MLP layers (MLP-GAN) and KAN layers (KAN-GAN). We compared the performance between those and found KAN-SGAN outperformed MLP-SGAN. We will publish our paper soon.

# Requirements 
* numpy==1.26.4
* numpyencoder==0.3.0
* torch==2.3.0+cu118
* torchvision==0.18.0+cu118
* tqdm==4.66.4

# Training

## Parameters
* *mode*: working mode ("train" or "test").
* *ds_name*: dataset name ("mnist", "fashion_mnist", or "sl_mnist").
* *model_name*: type of models (mlp_gan, kan_gan, mlp_kan_gan, kan_mlp_gan)
* *epochs*: the number of epochs.
* *batch_size*: the training batch size.
* *n_input*: The number of input neurons.
* *n_hidden*: The number of hidden neurons. We use only 1 hidden layer. You can modify the code (run.py) for more layers.
* *n_output*: The number of output neurons (classes). For MNIST, there are 10 classes.
* *grid_size*: The size of grids (default: 5). Use with bsrbf_kan and efficient_kan.
* *spline_order*: The order of spline (default: 3). Use with bsrbf_kan and efficient_kan.
* *num_grids*: The number of grids, equals grid_size + spline_order (default: 8). Use with fast_kan and faster_kan.
* *device*: use "cuda" or "cpu".
* *note*: write notes to save in model file names
* *n_latent*: latent dimension for the generator (default: 64)
* *n_examples*: the number of examples in the training set used for training (default: -1, mean use all training data)
* *kan_layer*: kan layers used in the discriminator and the generator

## Commands

# Acknowledgements

# Paper
We will publish our paper soon.

# Contact
If you have any questions, please contact: tahoangthang@gmail.com. If you want to know more about me, please visit website: https://tahoangthang.com.
