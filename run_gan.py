import argparse
import math
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import itertools

from file_io import *
from models import MLP_Discriminator, MLP_Generator, KAN_Generator, KAN_Discriminator
from pathlib import Path
from prettytable import PrettyTable
from scipy.spatial import distance
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
   
def run(model_name = 'kan_gan', batch_size = 64, n_input = 28*28, epochs = 10, n_output = 10, n_hidden = 64, \
        grid_size = 5, num_grids = 8, spline_order = 3, ds_name = 'mnist', n_examples = -1, note = '0', n_latent = 64, kan_layer = 'bsrbf_kan'):

    """
        note: short description 
    """    
    start = time.time()

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    
    # load datasets
    trainset, valset = [], []
    if (ds_name == 'mnist'):
        trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        valset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    elif(ds_name == 'fashion_mnist'):
        trainset = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        valset = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    elif(ds_name == 'sl_mnist'):
        from ds_model import SignLanguageMNISTDataset
        trainset = SignLanguageMNISTDataset(csv_file='data\SignMNIST\sign_mnist_train.csv', transform=transform)
        valset = SignLanguageMNISTDataset(csv_file='data\SignMNIST\sign_mnist_test.csv', transform=transform)
    
    if (n_examples != -1):
        trainset = torch.utils.data.Subset(trainset, range(n_examples))

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    # Create model storage
    output_path = 'output/'
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    if (model_name == 'mlp_gan'):
        saved_model_name = model_name + '__' +  ds_name + '__' + note + '.pth'
        saved_model_history =  model_name + '__' + ds_name + '__' + note + '.json'
    else:
        saved_model_name = model_name + '__' + kan_layer  + '__' +  ds_name + '__' + note + '.pth'
        saved_model_history =  model_name + '__' + kan_layer + '__' + ds_name + '__' + note + '.json'
    with open(os.path.join(output_path, saved_model_history), 'w') as fp: pass

    # Define model
    G, D = {}, {}
    print('model_name: ', model_name)
    if (model_name == 'mlp_gan'):
        # Initialize generator and discriminator
        G = MLP_Generator([n_latent, n_hidden, n_input]).to(device)
        D = MLP_Discriminator([n_input, n_hidden, n_output]).to(device)

    elif(model_name == 'kan_gan'):
        G = KAN_Generator([n_latent, n_hidden, n_input], grid_size = grid_size, spline_order = spline_order,    \
            num_grids = num_grids, kan_layer = kan_layer).to(device)
        D = KAN_Discriminator([n_input, n_hidden, n_output], grid_size = grid_size, spline_order = spline_order, \
            num_grids = num_grids, kan_layer = kan_layer).to(device)
        
    elif(model_name == 'mlp_kan_gan'):   
        G = MLP_Generator([n_latent, n_hidden, n_input]).to(device)
        D = KAN_Discriminator([n_input, n_hidden, n_output], grid_size = grid_size, spline_order = spline_order, \
            num_grids = num_grids, kan_layer = kan_layer).to(device)
    
    elif(model_name == 'kan_mlp_gan'):   
        G = KAN_Generator([n_latent, n_hidden, n_input], grid_size = grid_size, spline_order = spline_order, \
            num_grids = num_grids, kan_layer = kan_layer).to(device)
        D = MLP_Discriminator([n_input, n_hidden, n_output]).to(device)
    
    print('Discriminator parameters: ', count_parameters(D))
    print('Generator parameters: ', count_parameters(G))
    
    # Loss functions
    adversarial_loss = nn.BCELoss()
    auxiliary_loss = nn.CrossEntropyLoss()
    #sim_loss = torch.nn.MSELoss()

    # Optimizers
    d_optimizer = optim.Adam(D.parameters(), lr=1e-3, weight_decay=1e-4)
    g_optimizer = optim.Adam(G.parameters(), lr=1e-3, weight_decay=1e-4)
    d_scheduler = optim.lr_scheduler.ExponentialLR(d_optimizer, gamma=0.8)
    g_scheduler = optim.lr_scheduler.ExponentialLR(g_optimizer, gamma=0.8)
    
    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    best_epoch, best_accuracy = 0, 0

    y_true = [labels.tolist() for images, labels in valloader]
    y_true = sum(y_true, [])

    for epoch in range(1, epochs + 1):
        # Train
        D.train()
        G.train()
        train_accuracy, fake_accuracy = 0, 0
        train_g_loss, train_d_loss = 0, 0
        real_adv_ave, fake_adv_ave = 0, 0
        with tqdm(trainloader) as pbar:
            for i, (images, labels) in enumerate(pbar):
                
                # Prepare real images and labels
                images = images.view(-1, n_input).to(device)
                #images = images.reshape(n_input, -1).to(device)
                labels = labels.to(device)
                
                # Generate fake images       
                z = torch.randn(batch_size, n_latent).to(device)
                fake_images = G(z)

                # Train discriminator
                real_adv, real_aux = D(images.detach())
                real_adv_ave += real_adv.mean().item()
                  
                # Make the same size for last batch
                if (real_adv.shape[0] != real_labels.shape[0]):
                    real_labels = torch.ones(real_adv.shape[0], 1).to(device)
                
                d_loss_real_adv = adversarial_loss(real_adv, real_labels)
                d_loss_real_aux = auxiliary_loss(real_aux, labels)
                d_loss_real = d_loss_real_adv + d_loss_real_aux
                
                # Calculate training accuracy
                train_accuracy += (real_aux.argmax(dim=1) == labels.to(device)).float().mean().item()

                fake_adv, fake_aux = D(fake_images.detach())
                '''if (fake_adv.shape[0] > fake_labels.shape[0]):
                    fake_adv = fake_adv[:fake_labels.shape[0], :] 
                elif(fake_adv.shape[0] < fake_labels.shape[0]):
                    fake_labels = torch.zeros(fake_adv.shape[0], 1).to(device)'''
                d_loss_fake_adv = adversarial_loss(fake_adv, fake_labels)
                '''if (fake_aux.shape[0] != labels.shape[0]):
                    fake_aux = fake_aux[:labels.shape[0], :]
                d_loss_fake_aux = auxiliary_loss(fake_aux, labels)
                d_loss_fake = d_loss_fake_adv + d_loss_fake_aux'''
                
                d_loss = d_loss_real + d_loss_fake_adv
                train_d_loss += d_loss.item()
                
                d_optimizer.zero_grad()
                d_loss.backward()
                nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
                d_optimizer.step()
                
                # Train generator
                fake_adv, fake_aux = D(fake_images.detach())
                fake_adv_ave += fake_adv.mean().item()

                # make the same size for last batch
                if (fake_adv.shape[0] != real_labels.shape[0]):
                    real_labels = torch.ones(fake_adv.shape[0], 1).to(device)
                g_loss_adv = adversarial_loss(fake_adv, real_labels)

                if (fake_aux.shape[0] != labels.shape[0]):
                    fake_aux = fake_aux[:labels.shape[0], :]
                g_loss_aux = auxiliary_loss(fake_aux, labels)
                
                '''if (fake_images.shape[0] != images.shape[0]):
                    fake_images = fake_images[:images.shape[0], :]
                g_sim_loss = sim_loss(fake_images, images)'''
  
                g_loss = g_loss_aux + g_loss_adv
                train_g_loss += g_loss.item()
                
                fake_labels_pred = fake_aux.argmax(dim=1)
                fake_accuracy += (fake_labels_pred == labels).float().mean().item()
                
                g_optimizer.zero_grad()
                g_loss.backward()
                nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
                g_optimizer.step()

                pbar.set_postfix(train_accuracy=train_accuracy/len(trainloader), lr=d_optimizer.param_groups[0]['lr'])
        
        # Incorporate unlabelled data for discriminator training
        '''with tqdm(valloader) as pbar:
            for i, (images, _) in enumerate(pbar):
            #for images, _ in valloader: 
                images = images.view(-1, n_input).to(device)

                # Generate fake images       
                z = torch.randn(batch_size, n_latent).to(device)
                fake_images = G(z)

                # Train discriminator with unlabelled data
                real_adv, _ = D(images.detach())
                d_loss_unlabelled_adv = adversarial_loss(real_adv, real_labels[:real_adv.shape[0]])

                fake_adv, _ = D(fake_images.detach())
                d_loss_fake_unlabelled_adv = adversarial_loss(fake_adv, fake_labels[:fake_adv.shape[0]])

                d_loss_unlabelled = d_loss_unlabelled_adv + d_loss_fake_unlabelled_adv
                train_d_loss += d_loss_unlabelled.item()

                d_optimizer.zero_grad()
                d_loss_unlabelled.backward()
                nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
                d_optimizer.step()
                
                # Train generator with unlabelled data
                fake_adv, fake_aux = D(fake_images.detach())

                if (fake_adv.shape[0] != real_labels.shape[0]):
                    real_labels = torch.ones(fake_adv.shape[0], 1).to(device)
                g_loss_adv = adversarial_loss(fake_adv, real_labels)

                if (fake_aux.shape[0] != labels.shape[0]):
                    fake_aux = fake_aux[:labels.shape[0], :]
                g_loss_aux = auxiliary_loss(fake_aux, labels)
                g_loss_unlabelled = g_loss_aux + g_loss_adv
                train_g_loss += g_loss_unlabelled.item()

                g_optimizer.zero_grad()
                g_loss_unlabelled.backward()
                nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
                g_optimizer.step()
                
                pbar.set_postfix(lr=d_optimizer.param_groups[0]['lr'])'''
            
        
        g_scheduler.step()
        d_scheduler.step()
        
        train_accuracy = train_accuracy/ len(trainloader)
        fake_accuracy = fake_accuracy/ len(trainloader)
        
        # Compute validation accuracy
        D.eval()
        G.eval()
        val_loss, val_accuracy = 0, 0
        y_pred = []
        
        with torch.no_grad():
            for images, labels in valloader:
                images = images.view(-1, n_input).to(device)
                labels = labels.to(device)
                _, outputs = D(images)
                _, predicted = torch.max(outputs.data, 1)

                val_accuracy += ((predicted == labels).float().mean().item())
                y_pred += [x.item() for x in predicted]
                val_loss += auxiliary_loss(outputs, labels.to(device)).item()
        D.train()
        G.train()

        # calculate F1, Precision and Recall
        #f1 = f1_score(y_true, y_pred, average='micro')
        #pre = precision_score(y_true, y_pred, average='micro')
        #recall = recall_score(y_true, y_pred, average='micro')
        
        f1 = f1_score(y_true, y_pred, average='macro')
        pre = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')

        val_loss /= len(valloader)
        val_accuracy /= len(valloader)

        # Choose best model
        if (val_accuracy > best_accuracy):
            best_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(D, output_path + '/' + saved_model_name)
        fake_accuracy
        
        print(f'Epoch [{epoch}/{epochs}], Discriminator Accuracy: {train_accuracy:.6f}, Discriminator Loss: {train_d_loss/len(trainloader):.6f}, Generator Accuracy: {fake_accuracy:.6f}, Generator Loss: {train_g_loss/len(trainloader):.6f}, D(x): {real_adv_ave/len(trainloader):.6f}, D(G(z)): {fake_adv_ave/len(trainloader):.6f}')
        #print(f'Epoch [{epoch}/{epochs}], Discriminator Accuracy: {train_accuracy:.6f}, Discriminator Loss: {train_d_loss/len(trainloader):.6f}, Generator Loss: {train_g_loss/len(trainloader):.6f}')         
        print(f"Epoch [{epoch}/{epochs}], Val Accuracy: {val_accuracy:.6f}, Val Loss: {val_loss:.6f}, F1: {f1:.6f}, Recall: {recall:.6f}, Precision: {pre:.6f} ")
        
        write_single_dict_to_jsonl_file(output_path + '/' + saved_model_history, {'epoch':epoch, 'val_accuracy':val_accuracy, 'train_accuracy':train_accuracy, 'f1_macro':f1, 'pre_macro':pre, 're_macro':recall, 'best_epoch':best_epoch, 'val_loss': val_loss, 'train_d_loss':train_d_loss, 'train_g_loss':train_g_loss}, file_access = 'a')
        
        # print the first batch of fake images
        #fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
        #vutils.save_image(fake_images.data, f'fake_images_{epoch}.png')
    
    end = time.time()
    print(f"Training time (s): {end-start}")
    write_single_dict_to_jsonl_file(output_path + '/' + saved_model_history, {'training time':end-start}, file_access = 'a')    
    
def main(args):
    if (args.mode == 'train'):
        run(model_name = args.model_name, batch_size = args.batch_size, epochs = args.epochs, \
            n_input = args.n_input, n_output = args.n_output, n_hidden = args.n_hidden, \
            grid_size = args.grid_size, num_grids = args.num_grids, spline_order = args.spline_order, ds_name = args.ds_name, n_examples =args.n_examples, note = args.note, n_latent = args.n_latent, kan_layer = args.kan_layer)
                    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training Parameters')
    parser.add_argument('--mode', type=str, default='train') # or test
    parser.add_argument('--model_name', type=str, default='efficient_kan')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_input', type=int, default=28*28)
    parser.add_argument('--n_hidden', type=int, default=64)
    parser.add_argument('--n_output', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_path', type=str, default='output/model.pth') # only for testing
    parser.add_argument('--grid_size', type=int, default=5)
    parser.add_argument('--num_grids', type=int, default=8)
    parser.add_argument('--spline_order', type=int, default=3)
    parser.add_argument('--ds_name', type=str, default='mnist')
    parser.add_argument('--note', type=str, default='full')
    parser.add_argument('--n_latent', type=int, default=64)
    parser.add_argument('--n_examples', type=int, default=-1)
    parser.add_argument('--kan_layer', type=str, default='bsrbf_kan')
    
    args = parser.parse_args()
    
    global device
    device = args.device
    if (args.device == 'cuda'): # check available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    main(args)
    
#python run_gan.py --mode "train" --model_name "kan_gan" --epochs 35 --batch_size 64 --n_latent 64 --n_input 784 --n_hidden 64 --n_output 10 --grid_size 5 --spline_order 3 --ds_name "fashion_mnist" --note "full" --kan_layer "bsrbf_kan" 

#python run_gan.py --mode "train" --model_name "kan_gan" --epochs 25 --batch_size 64 --n_latent 64 --n_input 784 --n_hidden 64 --n_output 10 --grid_size 5 --spline_order 3 --ds_name "fashion_mnist" --note "full" --kan_layer "efficient_kan"

#python run_gan.py --mode "train" --model_name "kan_gan" --epochs 25 --batch_size 64 --n_latent 64 --n_input 784 --n_hidden 64 --n_output 10 --num_grids 8 --ds_name "fashion_mnist" --note "full" --kan_layer "fast_kan"

#python run_gan.py --mode "train" --model_name "kan_gan" --epochs 35 --batch_size 64 --n_latent 64 --n_input 784 --n_hidden 64 --n_output 10 --num_grids 8 --ds_name "fashion_mnist" --note "full" --kan_layer "faster_kan"

#python run_gan.py --mode "train" --model_name "mlp_gan" --epochs 35 --batch_size 64 --n_latent 64 --n_input 784 --n_hidden 64 --n_output 10 --ds_name "fashion_mnist" --note "full"