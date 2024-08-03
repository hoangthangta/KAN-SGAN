**Updating code...**

# KAN-SGAN
Semi-supervised learning with Generative Adversarial Networks (SGANs) using Kolmogorov-Arnold Network Layers (KANLs). In this repo, we set up GANs using MLP layers (MLP-GAN) and KAN layers (KAN-GAN).

# Requirements 
* numpy==2.0.1
* numpyencoder==0.3.0
* prettytable==3.10.2
* scikit_learn==1.4.2
* scipy==1.14.0
* torch==2.3.1
* torchvision==0.18.1
* tqdm==4.66.4
* transformers==4.40.1

# Training
We trained the models  **on GeForce RTX 3060 Ti** (with other default parameters; see Commands).

## Parameters
* *mode*: working mode ("train").
* *ds_name*: dataset name ("mnist", "fashion_mnist", or "sl_mnist" (SignMNIST)).
* *model_name*: type of models (**mlp_gan, kan_gan, mlp_kan_gan, kan_mlp_gan**). If using **mlp_kan_gan**, the generator is mlp and the discriminator is kan (defined in kan_layer). If using **kan_mlp_gan**, the generator is kan (defined in kan_layer) and the discriminator is mlp.
* *epochs*: the number of epochs.
* *batch_size*: the training batch size.
* *n_input*: The number of input neurons.
* *n_hidden*: The number of hidden neurons. We use only 1 hidden layer. You can modify the code (run.py or run_gan.py) for more layers.
* *n_output*: The number of output neurons (classes). For MNIST, there are 10 classes.
* *grid_size*: The size of grids (default: 5). Use with bsrbf_kan and efficient_kan.
* *spline_order*: The order of spline (default: 3). Use with bsrbf_kan and efficient_kan.
* *num_grids*: The number of grids, equals grid_size + spline_order (default: 8). Use with fast_kan and faster_kan.
* *device*: use "cuda" or "cpu".
* *note*: write notes to save in model file names
* *n_latent*: latent dimension for the generator (default: 64)
* *n_examples*: the number of examples in the training set used for training (default: -1, means all training data)
* *kan_layer*: kan layers used in the discriminator and the generator (**efficient_kan, fast_kan, faster_kan, bsrbf_kan**)

## Commands
### Fashion MNIST
```python run_gan.py --mode "train" --model_name "kan_gan" --epochs 25 --batch_size 64 --n_latent 64 --n_input 784 --n_hidden 64 --n_output 10 --grid_size 5 --spline_order 3 --ds_name "fashion_mnist" --note "full_0" --kan_layer "bsrbf_kan" --n_examples 1000```

```python run_gan.py --mode "train" --model_name "kan_gan" --epochs 25 --batch_size 64 --n_latent 64 --n_input 784 --n_hidden 64 --n_output 10 --grid_size 5 --spline_order 3 --ds_name "fashion_mnist" --note "full_0" --kan_layer "efficient_kan" --n_examples 1000```

```python run_gan.py --mode "train" --model_name "kan_gan" --epochs 25 --batch_size 64 --n_latent 64 --n_input 784 --n_hidden 64 --n_output 10 --num_grids 8 --ds_name "fashion_mnist" --note "full_0" --kan_layer "fast_kan" --n_examples 1000```

```python run_gan.py --mode "train" --model_name "kan_gan" --epochs 25 --batch_size 64 --n_latent 64 --n_input 784 --n_hidden 64 --n_output 10 --num_grids 8 --ds_name "fashion_mnist" --note "full_0" --kan_layer "faster_kan" --n_examples 1000```

```python run_gan.py --mode "train" --model_name "mlp_gan" --epochs 25 --batch_size 64 --n_latent 64 --n_input 784 --n_hidden 64 --n_output 10 --ds_name "fashion_mnist" --note "full_0" --n_examples 1000```

# Experiments
## FashionMNIST
We train all models over 35 epochs with full training data. For the MLP combined with KANs, the model architecture is configured as (768, 64, 10). In the case of GANs, the discriminator is designed with an architecture of (768, 64, 10), while the generator has an architecture of (64, 64, 768). The results presented are based on a single random training session.

# Acknowledgements
There is a lot of work we need to say thank you here; update later.
* EfficientKAN: https://github.com/Blealtan/efficient-kan
* FastKAN: https://github.com/ZiyaoLi/fast-kan
* FasterKAN: https://github.com/AthanasiosDelis/faster-kan
* BSRBF-KAN: https://github.com/hoangthangta/BSRBF_KAN (also our previous work)
* Original KAN: https://github.com/KindXiaoming/pykan

# Paper
We will publish our paper soon.

# Contact
If you have any questions, please contact: tahoangthang@gmail.com. If you want to know more about me, please visit website: https://tahoangthang.com.
