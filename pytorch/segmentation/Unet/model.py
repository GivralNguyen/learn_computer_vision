import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = [] #1,64,160,160 / 1,128,80,80 /1,256,40,40 / 1,512,2020
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            #1,1,160,160 -> 1,64,160,160 -> 1,64,80,80 ->1,128,80,80->1,128,40,40->1,256,40,40->1,256,20,20->1,512,20,20->1,512,10,10
        x = self.bottleneck(x) #1,1024,10,10
        skip_connections = skip_connections[::-1] # reverse skip 

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x) #1,512,20,20  / 1,256,40,40 / 1,128,80,80 / 1,64,160,160
            skip_connection = skip_connections[idx//2] #1,512,20,20 / 1,256,40,40  / 1,128,80,80 /1,64,160,160

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1) #1,1024,20,20 / 1,512,40,40 / 1,256,80,80 /1,128,160,160
            x = self.ups[idx+1](concat_skip) #1,512,20,20 /1,256,40,40 / 1,256,80,80 /1,64,160,160

        return self.final_conv(x) #1,1,160,160

def test():
    x = torch.randn((1, 1, 160, 160))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    dummy_input = x

    torch.onnx.export(model, dummy_input, "mbv2_ssdlite.onnx")
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()