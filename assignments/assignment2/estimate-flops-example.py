import torch
from torchvision.models import resnet50,vgg16
from yolov3.yolo import YOLOv3
from pthflops import count_ops

# Create a network and a corresponding input
device = 'cuda:0'
detector = YOLOv3(device='cuda', img_size=1024)
inp = torch.rand(1,3,1024,1024).to(device)

# Count the number of FLOPs
count_ops(detector, inp)