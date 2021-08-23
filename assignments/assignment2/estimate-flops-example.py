import torch
from torchvision.models import resnet18

from pthflops import count_ops

# Create a network and a corresponding input
device = 'cuda:0'
model = resnet18().to(device)
inp = torch.rand(1,3,224,224).to(device)

# Count the number of FLOPs
count_ops(model, inp)