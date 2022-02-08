
'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import sys
import time

sys.path.insert(0,'/home/quannm/code/learn_computer_vision/pytorch/classification')
from models import *
from utils import progress_bar
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--batchsize', default=32, type=float, help='batch size')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='/home/quannm/code/learn_computer_vision/pytorch/dataset', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='/home/quannm/code/learn_computer_vision/pytorch/dataset', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batchsize, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')

# model_name = "VGG19" # 93.270 30s804ms 2014 0.4/0.44 GFLOPs
# net = VGG('VGG19') 

# model_name = "ResNet18" # 95.05 , 33s500ms 2015  0.56/0.64 GFLOPs (32 3 32 32)
# net = ResNet18()

# model_name = "MobileNetv2" # 94.810 1m 23s 0.31 GFLOPs/0.58 GFLOPs (32, 3, 32, 32)
# net = MobileNetV2(10, alpha = 1)

# model_name = "PreActResNet18" # 94.740 0.56/06.3 GFLOPS
# net = PreActResNet18()

# model_name = "GhostVGG19" # 
# net = GhostVGG('VGG19')

# model_name = "GhostResnet18" # 
# net = GhostResNet18()

# model_name = "Shufflenetv2_1.5" #  58s187 1.5 0.1 GFLOPs/ 0.19 GFLOPs
# net = ShuffleNetV2(net_size=1.5)


# model_name = "densenet_cifar" # 1m 50s 0.13 GFLOPs/0.36 GFLOPs
# net = densenet_cifar()

model_name = "SENet" # 41s621 0.56 GFLOPs/0.65 GFLOPs
net = SENet18()

# model_name = "EfficientnetV2" # error?
# net = effnetv2_s()

# model_name = "MobileNet" #0.05/ 0.10 GFLOPS 
# net = MobileNet()



# model_name = "DPN92" # 2.07 GFLOPs/2.86 GFLOPs
# net = DPN92()


# model_name = "EfficientNetB0" #0.03 GFLOPs/ 0.09 GFLOPs 
# ! EFFICIENTNET IS DIFFICULT TO TRAIN
"""
 not so much faster to run on e.g. NVIDIA GPUs, where much lower GFLOPs don't necessarily translate into much lower milliseconds 
 due to EfficientNet having _a ton_ more ops in the graph, and those ops being "small" and unable to take advantage of all the throughput. 
 Indeed, some configs are slower than comparably accurate alternatives. 
 TL;DR: just because Google is able to get sexy results out of this doesn't mean 
 there aren't better options available to you to solve practical tasks.
 https://news.ycombinator.com/item?id=25040917
"""
# net = EfficientNetB0() 

# model_name = "GhostNet" # 0.01 GFLOPs bs32
# net = ghost_net()



# model_name = "GhostNetRes18" # 0.29 GFLOPs/0.43 GFLOPs bs32
# net = ghost_resnet18()

# model_name = "Shufflenet" #0.04 GFLOPs/0.15 GFLOPs
# net = ShuffleNetG2()



# model_name = "Googlenet" # 1.55 GFLOPs/1.86 GFLOPs
# net = GoogLeNet()

# model_name = "Lenet"  #0.00 GFLOPs xD
# net = LeNet()

# model_name = "PnasNet" #0.08 GFLOPs/0.27 GFLOPs
# net = PNASNetB()

# model_name = "Regnet" #0.23 GFLOPs/0.43 GFLOPs
# net = RegNetX_200MF()

# model_name = "Resnext" #1.42 GFLOPs/1.88 GFLOPs
# net = ResNeXt29_2x64d()



# model_name = "simplifiedDLA" #/0.92 GFLOPs/1.06
# net = SimpleDLA()

# model_name = "DLA" #1.04/1.19
# net = DLA()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
weight_decay = 4e-5
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
writer = SummaryWriter(
            f"runs/CIFAR/{model_name}_MiniBatchSize_{args.batchsize}_LR_{args.lr}_decay_{weight_decay}"
        )

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        writer.add_scalar("Training loss", train_loss/(batch_idx+1), global_step=epoch) #1
        writer.add_scalar(
            "Training Accuracy", 100.*correct/total, global_step=epoch 
        )
                

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            writer.add_scalar("Test loss", test_loss/(batch_idx+1), global_step=epoch) #1
            writer.add_scalar(
                "Test Accuracy", 100.*correct/total, global_step=epoch 
            )
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()