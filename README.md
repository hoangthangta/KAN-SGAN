**Updating code...**

# KAN-SGAN
Semi-supervised learning with Generative Adversarial Networks (SGANs) using Kolmogorov-Arnold Network Layers (KANLs)

In this repo, we set up GANs using MLP layers (MLP-GAN) and KAN layers (KAN-GAN). We will soon publish our paper.

# Requirements 
* numpy==1.26.4
* numpyencoder==0.3.0
* torch==2.3.0+cu118
* torchvision==0.18.0+cu118
* tqdm==4.66.4

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
* *n_examples*: the number of examples in the training set used for training (default: -1, mean use all training data)
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
We train all models over 35 epochs. For the MLP combined with KANs, the model architecture is configured as (768, 64, 10). In the case of GANs, the discriminator is designed with an architecture of (768, 64, 10), while the generator has an architecture of (64, 64, 768). The results presented are based on a single random training session.

 Network | Total Layers | Training Accuracy | Val Accuracy | Macro F1 | Macro Precision | Macro Recall | Training time (seconds) | Params
 | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
 | mlp | 2 (768, 64, 10) | 93.59 | 88.83 | 88.80 | 88.83 | 88.84 | 207 | - |
 | mlp_gan | 2 (768, 64, 10) | 93.97 | 88.88 | 88.85 | 88.83 | 88.89 | 319 | - |
 | faster_kan | 2 (768, 64, 10) | 94.36 | 89.12 | 89.05 | 89.05 | 89.10 | 223 | - |
 | kan_gan (faster_kan) | 2 (768, 64, 10) | 92.87 | 89.29 | 89.21 | 89.20 | 89.27 | 328 | - |
 | fast_kan | 2 (768, 64, 10) | 98.19 | 89.37 | 89.32 | 89.32 | 89.35 | 207 | - |
 | kan_gan (fast_kan) | 2 (768, 64, 10) | 97.71 | 89.39 | 89.35 | 89.35 | 89.37 | 360 | - |
 | efficient_kan | 2 (768, 64, 10) | 95.11 | 89.03 | 88.97 | 88.97 | 89.01 | 255 | - |
 | kan_gan (efficient_kan) | 2 (768, 64, 10) | 92.23  | 88.13 | 88.08 | 88.07 | 88.14 | 468 | - |
 | bsrbf_kan | 2 (768, 64, 10) | **99.32** | 89.11 | 89.07 | 89.07 | 89.09 | 281 | - |
 | kan_gan (bsrbf_kan) | 2 (768, 64, 10) |  99.14 | **89.44**| **89.41** | **89.41** | **89.45** | 589 | - |

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
