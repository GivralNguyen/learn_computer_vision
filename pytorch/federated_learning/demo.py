###############################
##### importing libraries #####
###############################

import os
import random
from tqdm import tqdm
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset   
torch.backends.cudnn.benchmark=True
import sys
sys.path.insert(0,'/home/quannm/code/learn_computer_vision/pytorch/classification/models')
from vgg_classifier import VGG
from resnet_classifier import BasicBlock,Bottleneck,ResNet
from ghostnet_classifier import *
import time

##### Hyperparameters for federated learning #########
num_clients = 20 # total number of clients 
num_selected = 6 # number of clients selected per round 
num_rounds = 150 # number of communication rounds 
epochs = 5 #number of epoch per round 
batch_size = 32 # batch size 
criterion = nn.CrossEntropyLoss()
#############################################################
##### Creating desired data distribution among clients  #####
#############################################################

# Image augmentation

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Loading CIFAR10 using torchvision.datasets
traindata = datasets.CIFAR10('/home/quannm/code/learn_computer_vision/pytorch/dataset', train=True, download=True,
                       transform= transform_train)

print("Information about train dataset:")
print(traindata)

# Dividing the training data into num_clients, with each client having equal number of images
traindata_split = torch.utils.data.random_split(traindata, [int(traindata.data.shape[0] / num_clients) for _ in range(num_clients)])

print("Data split among "+ str(num_clients)+ " clients, each client has "+ str(int(traindata.data.shape[0] / num_clients)) +" data points")

# Creating a pytorch loader for a Deep Learning model
train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]


# Normalizing the test images
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_data = datasets.CIFAR10('/home/quannm/code/learn_computer_vision/pytorch/dataset', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        )
print("Information about test data: ")
print(test_data)

# Loading the test iamges and thus converting them into a test_loader
test_loader = torch.utils.data.DataLoader(test_data
        , batch_size=batch_size, shuffle=True)


def client_update(client_model, optimizer, train_loader, epoch=5):
    """
    This function updates/trains client model on client data
    """
    model.train()
    for e in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda() #32,3,32,32 / 32 
            optimizer.zero_grad()
            output = client_model(data) #32,10
            loss = criterion(output, target) # value 
            loss.backward()
            optimizer.step()
    return loss.item()

def server_aggregate(global_model, client_models):
    """
    This function has aggregation method 'mean'
    """
    ### This will take simple mean of the weights of models ###
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())

def test(global_model, test_loader):
    """This function test the global model on test data and returns test loss and test accuracy """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = global_model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    return test_loss, acc

############################################
#### Initializing models and optimizer  ####
############################################

#### global model ##########
# global_model =  VGG('VGG19').cuda()
# global_model = ResNet(Bottleneck, [3, 4, 6, 3]).cuda()
global_model = ghost_net().cuda()
############## client models ##############
client_models = [ ghost_net().cuda() for _ in range(num_selected)]
# client_models = [ VGG('VGG19').cuda() for _ in range(num_selected)]
# client_models = [ResNet(Bottleneck, [3, 4, 6, 3]).cuda() for _ in range(num_selected)]
for model in client_models:
    model.load_state_dict(global_model.state_dict()) ### initial synchronizing with global model 

############### optimizers ################
opt = [optim.SGD(model.parameters(), lr=0.1) for model in client_models]


###### List containing info about learning #########
losses_train = []
losses_test = []
acc_train = []
acc_test = []
# Runnining FL

for r in range(num_rounds):
    # select random clients
    start = time.time()
    client_idx = np.random.permutation(num_clients)[:num_selected] # array of client ids 
    # client update
    loss = 0
    for i in tqdm(range(num_selected)):
        loss += client_update(client_models[i], opt[i], train_loader[client_idx[i]], epoch=epochs) # run 5 epochs on each selected client 
    
    losses_train.append(loss)
    # server aggregate
    server_aggregate(global_model, client_models)
    
    test_loss, acc = test(global_model, test_loader)
    losses_test.append(test_loss)
    acc_test.append(acc)
    print('%d-th round' % r)
    print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (loss / num_selected, test_loss, acc))
    end = time.time()
    print("Time of one round: "+ str(end - start))