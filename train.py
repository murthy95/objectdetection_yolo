import torch
from torchsummary import summary
from yolov1 import yolonet

#make a network object
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = yolonet()
net = net.to(device)
print(summary(net, (3,448,448))

