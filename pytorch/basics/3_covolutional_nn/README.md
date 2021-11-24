# Simple Convolutional Neural Network

![CNN](https://miro.medium.com/max/2000/1*vkQ0hXDaQv57sALXAJquxA.jpeg)

```
-Main flow of CNN:

def forward(self, x):
        #torch.Size([64, 1, 28, 28]) #input image
        x = self.conv1(x)
        #torch.Size([64, 8, 28, 28]) # first convolution 
        x = F.relu(x)
        #torch.Size([64, 8, 28, 28]) # relu 
        x = self.pool(x)
        #torch.Size([64, 8, 14, 14]) # pooling 
        x = self.conv2(x)
        #torch.Size([64, 16, 14, 14]) # second convolution
        x = F.relu(x)
        #torch.Size([64, 16, 14, 14]) # relu
        x = self.pool(x) 
        #torch.Size([64, 16, 7, 7]) # pooling
        x = x.reshape(x.shape[0], -1)  
        #torch.Size([64, 784])  # flatten
        x = self.fc1(x) 
        # torch.Size([64, 10]) # fully connected
        return x
```