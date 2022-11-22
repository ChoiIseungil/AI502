import os
import time
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt
from PIL import Image

from models import mlp
from models import vgg
from models import resnet
from models import vgg_dropout

PATH = './save/'

def load_dataset(batchsize):
    custom_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        #transforms.Grayscale(),
        #transforms.Lambda(lambda x: x/255.),
        transforms.ToTensor()
    ])

    train_dataset = ImageFolder(root='./tiny-imagenet-200/train',
                                transform=custom_transform)

    valid_dataset = ImageFolder(root='./tiny-imagenet-200/val',
                                transform=custom_transform)

    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batchsize,
                            shuffle=True,
                            num_workers=12)

    valid_loader = DataLoader(dataset=valid_dataset,
                            batch_size=batchsize,
                            shuffle=False,
                            num_workers=12)
    return train_loader, valid_loader

def load_model(model_name,regularization):
    if model_name=='mlp': return mlp.MLP()
    if model_name=='vgg': 
        if regularization=='dropout': return vgg_dropout.VGG16()
        else: return vgg.VGG16()
    if model_name=='resnet': return resnet.ResNet18()
    raise Exception("Wrong Model Name")

def load_optimizer(model,learning_rate,optimizer_name):
    if optimizer_name=='':          return torch.optim.SGD(model.parameters(), lr=learning_rate)
    if optimizer_name=='momentum':  return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    if optimizer_name=='adagrad':   return torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    if optimizer_name=='adam':   
        print("Adam optimizer Loaded")   
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    raise Exception("Wrong Optimizer")

def main():
    args = get_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model,args.r).to(device)
    train_loader, valid_loader = load_dataset(args.batchsize)
    torch.manual_seed(0)
    
    optimizer = load_optimizer(model,args.lr,args.o)

    loss = {}
    loss['train'] = []
    acc = {}
    acc['train'] = []
    acc['val'] = []

    def compute_accuracy(model, data_loader):
        correct_pred, num_examples = 0, 0
        for i, (features, targets) in enumerate(data_loader):
                
            features = features.to(device)
            targets = targets.to(device)

            logits, probas = model(features)
            _, predicted_labels = torch.max(probas, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
        return correct_pred.float()/num_examples * 100
        
    def train():

        start_time = time.time()
        for epoch in range(args.epoch):
            train_loss_ep = 0
            model.train()
            for batch_idx, (features, targets) in enumerate(train_loader):

                features = features.to(device)
                targets = targets.to(device)
                
                ### FORWARD AND BACK PROP
                logits, probas = model(features)
                cost = F.cross_entropy(logits, targets)
                if args.r=='l2': cost += 1e-4 * sum( [ (p ** 2).sum()  for p in model.parameters() ] )      #l2-lambda==1e-4
                if args.r=='l1': cost += 1e-5 * sum( [ (torch.abs(p)).sum() for p in model.parameters() ] ) #l1-lambda==1e-3
                optimizer.zero_grad()
                
                cost.backward()
                
                ### UPDATE MODEL PARAMETERS
                optimizer.step()
                
                ### LOGGING
                if not batch_idx % 50:
                    print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                        %(epoch+1, args.epoch, batch_idx, 
                            len(train_loader), cost))

            model.eval()
            with torch.set_grad_enabled(False): # save memory during inference
                train_accuracy = compute_accuracy(model, train_loader)
                valid_accuracy = compute_accuracy(model, valid_loader)
                print('Epoch: %03d/%03d | Train: %.3f%% | Valid: %.3f%%' % (
                    epoch+1, args.epoch, 
                    train_accuracy,
                    valid_accuracy))

            loss['train'].append(cost.item())
            acc['train'].append(train_accuracy.item())
            acc['val'].append(valid_accuracy.item())

            print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

        print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

    def evaluate():
        with torch.set_grad_enabled(False): # save memory during inference
            print('Validation accuracy: %.2f%%' % (compute_accuracy(model, valid_loader)))

    def plot(loss, acc):
        fig = plt.figure()
        ax0 = fig.add_subplot(121, title="Loss")
        ax1 = fig.add_subplot(122, title="Accuracy")
        ax0.plot(loss['train'],label="Training Loss")

        ax1.plot(acc['train'],label="Training Accuracy")
        ax1.plot(acc['val'],label="Validation Accuracy")
        fig.savefig(os.path.join('./graph',args.model+args.r+args.o+'.jpg'))

    train()
    torch.save(model.state_dict(),"./checkpoint/"+args.model+args.r+args.o+".ckpt")
    evaluate()
    plot(loss,acc)

def get_args():
    parser = argparse.ArgumentParser(description='AI502 Programming Assignment #1, by Seungil Lee')

    parser.add_argument('-epoch', '-e', default = 100, type = int,
                        help='number of epochs to be trained')
    parser.add_argument('-batchsize', '-b', default = 32, type = int,
                        help='batch size')
    parser.add_argument('-model', '-m', required = True, type= str,
                        help='Choose among mlp,vgg and resnet')
    parser.add_argument('-gpu', '-g', required = True, default = 0, type = int,
                        help='Choose GPU to be used')
    parser.add_argument('-lr', '-l', default = 5e-2, type = float,
                        help='Learning Rate')
    parser.add_argument('-r', default = '', type=str,
                        help='Choose among dropout, l1, l2')
    parser.add_argument('-o', default = '', type=str,
                        help='Choose among momentum, adagrad, adam')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()