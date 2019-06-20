import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn

n_boxes = 2
n_classes = 20
class conv_layer(nn.Module):
	def __init__(self, in_maps, out_maps, kernel_size, stride, padding):
		super(conv_layer, self).__init__()
		self.conv = nn.Conv2d(in_maps, out_maps, kernel_size,\
													 stride=stride, padding=padding )
		self.bn = nn.BatchNorm2d(out_maps)
		
	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		return F.relu(x)

class conv_pool_block(nn.Module):
  def __init__(self, kernel_sizes, in_maps, out_maps, n_repeat, pooling=False, conv_layer=conv_layer):
    assert len(kernel_sizes) == len(out_maps), 'Inconsistent kernel and filter lengths'
     
    super(conv_pool_block, self).__init__()
    self.layers = []
    for _ in range(n_repeat):
      self.layers.append(conv_layer(in_maps, out_maps[0], 
                                    kernel_sizes[0], stride=1, 
                                    padding = kernel_sizes[0]//2))
      for i in range(1, len(out_maps)):
        self.layers.append(conv_layer(out_maps[i-1], out_maps[i], 
                                    kernel_sizes[i], stride=1, 
                                    padding = kernel_sizes[i]//2))
    if pooling:
      self.layers.append(nn.MaxPool2d(2,2))
      
    self.model = nn.Sequential(*self.layers)
      
  def forward(self, x):
    return self.model(x)
    
class yolonet(nn.Module):
  def __init__(self):
    super(yolonet, self).__init__()
    self.model = nn.Sequential(conv_layer(3, 64, 7,stride=2, padding=3),
                              nn.MaxPool2d(2, stride=2), 
                              conv_pool_block([3], 64, [192], 1, pooling=True), 
                              conv_pool_block([1, 3, 1, 3], 192, 
                                  [128, 256, 256, 512],
                                  1, pooling=True),
                              conv_pool_block([1, 3], 512, 
                                  [256, 512], 4), 
                              conv_pool_block([1, 3], 512, 
                                  [512, 1024],
                                  1, pooling=True), 
                              conv_pool_block([1, 3], 1024, 
                                  [512, 1024], 2),
                              conv_pool_block([3], 1024, 
                                  [1024], 1), 
                              conv_layer(1024, 1024, 3, stride=2, padding=1),
                              conv_pool_block([3, 3], 1024, 
                                  [1024, 1024], 1))
    self.Linear1 = nn.Linear(49*1024, 4096)
    self.Linear2 = nn.Linear(4096, 49*(n_boxes*5 + n_classes))
   
  def forward(self, x):
    x = self.model(x)
    x = x.view(x.size(0), -1)
    x = self.Linear1(x)
    x = self.Linear2(x)
    return x.view((x.size(0), 30, 7, 7))

