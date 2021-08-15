# ASSIGNMENT 1: BUILD VGG-16 FROM SCRATCH
- Build VGG16 from scratch. Train the network on MNIST dataset.
- The model of VGG-16 is given as: 
<p align="center"><img width="100%" src="https://media.geeksforgeeks.org/wp-content/uploads/20200219152327/conv-layers-vgg16.jpg" /></p>
- The structure of the .pth file of the VGG-16 is given as follow, given there are 10 output classes. 
- (Note: Relu , pooling and dropout is only included in forward step, not directly in .pth file.)
- Remember to also add relu after each convolutional layers. Add 0.5 dropout after relu of fc6 and fc7

```
VGG16(
  (conv1_1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv1_2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2_1): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2_2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv3_1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv3_2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv3_3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv4_1): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv4_2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv4_3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv5_1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv5_2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv5_3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc6): Linear(in_features=25088, out_features=4096, bias=True)
  (fc7): Linear(in_features=4096, out_features=4096, bias=True)
  (fc8): Linear(in_features=4096, out_features=1000, bias=True)
)
```

- To test your model, successfully load the weight given [here](https://drive.google.com/file/d/1medFTyklQZMY4pH8I-k0WFoTpCtGDusA/view?usp=sharing) using your own custom model.