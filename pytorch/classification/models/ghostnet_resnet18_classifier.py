
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# GhostModule
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]

# 基础残差
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            # nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            GhostModule(inchannel, outchannel, kernel_size=3, stride=stride),
            # nn.BatchNorm2d(outchannel),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            GhostModule(outchannel, outchannel, kernel_size=3, stride=1,relu=False),
            # nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                # nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                GhostModule(inchannel, outchannel, kernel_size=1, stride=stride,relu=False),
                # nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out

# ResNet
class GhostResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(GhostResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            # nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            # nn.ReLU()
            GhostModule(3, 64, kernel_size=1, stride=1),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
from pthflops import count_ops
def GhostResNet18():
    return GhostResNet(ResidualBlock)
def test():
    net = GhostResNet18()
    x = torch.randn(32, 3, 32, 32)
    # x = torch.randn(1, 3, 32, 32)
    count_ops(net,x)
    y = net(x)
    print(y.size())


if __name__ == "__main__":
    test()