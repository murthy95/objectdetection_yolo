import torch 
import numpy as np
import torchvision

n_boxes = 2
n_classes = 20
class conv_layer(torch.nn.Module):
	def __init__(self, in_maps, out_maps, kernel_size, stride, padding):
		super(conv_layer, self).__init__()
		self.conv = torch.nn.Conv2d(in_maps, out_maps, kernel_size,\
													 stride=stride, padding=padding )
		self.bn = torch.nn.BatchNorm2d(out_maps)
		
	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		return torch.nn.functional.leaky_relu(x, negative_slope=0.1)

class conv_pool_block(torch.nn.Module):
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
      self.layers.append(torch.nn.MaxPool2d(2,2))
      
    self.model = torch.nn.Sequential(*self.layers)
      
  def forward(self, x):
    return self.model(x)
    
class yolonet(torch.nn.Module):
  def __init__(self):
    super(yolonet, self).__init__()
    self.model = torch.nn.Sequential(conv_layer(3, 64, 7,stride=2, padding=3),
                              torch.nn.MaxPool2d(2, stride=2), 
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
    self.Linear1 = torch.nn.Linear(49*1024, 4096)
    self.Linear2 = torch.nn.Linear(4096, 49*(n_boxes*5 + n_classes))
   
  def forward(self, x):
    x = self.model(x)
    x = x.view(x.size(0), -1)
    x = self.Linear1(x)
    x = self.Linear2(x)
    return x.view((x.size(0), 30, 7, 7))

def transform_input(image):
	transform = torchvision.transforms.Compose(
				[torchvision.transforms.Resize((448, 448), interpolation=2),
					torchvision.transforms.ToTensor()])
  #torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
   #                    std=[0.229, 0.224, 0.225])
	transformed_image = transform(image)
	return transformed_image
      
def IoU(bb1, bb2):
	'''
	bb1 : input bounding box 1 the coordinates corresponds to 
	      center_x center_y width height
	bb2 : bounding box 2 coordinates in the same order as bb1
	'''
	minx = min(bb1[0] - bb1[2] / 2, bb2[0] - bb2[2] / 2)
	maxx = max(bb1[0] + bb1[2] / 2, bb2[0] + bb2[2] / 2)
	miny = min(bb1[1] - bb1[3] / 2, bb2[1] - bb2[3] / 2)
	maxy = max(bb1[1] + bb1[3] / 2, bb2[1] + bb2[3] / 2)
	diffx = (maxx - minx) - (bb1[2] + bb2[2]) 
	diffy = (maxy - miny) - (bb1[3] + bb2[3])
	if diffx < 0  and diffy < 0 :
		intersection = diffx * diffy
		return intersection / (bb1[2] * bb1[3] + bb2[2] * bb2[3] - intersection) 
	return 0

def non_max_suppression(predicted_tensor, obj_threshold=0.3, iou_threshold=0.6):
	#grid coordinates to add to predicted offset
	predicted_tensor = predicted_tensor.cpu()
	x = torch.tensor(torch.arange(7))
	y = torch.tensor(torch.arange(7))
	grid_x, grid_y = torch.meshgrid(x, y)
	box1, box2, classes = predicted_tensor.split([5, 5, 20], 1)
	box1 = torch.sigmoid(box1)
	box2 = torch.sigmoid(box2)
	classes = torch.argmax(classes, 1, keepdim=True).type(torch.float)
	grid = torch.cat([grid_x.unsqueeze(0), grid_y.unsqueeze(0)], 0).type(torch.float)
	box1_coords, box1_dims, box1_confs = box1.split([2,2,1], 1)
	box2_coords, box2_dims, box2_confs = box2.split([2,2,1], 1)
	box1_coords = (box1_coords + grid)/7
	box2_coords = (box2_coords + grid)/7
	box1 = torch.cat([box1_coords, box1_dims, box1_confs, classes], dim=1).permute(0, 2, 3, 1)
	box2 = torch.cat([box2_coords, box2_dims, box2_confs, classes], dim=1).permute(0, 2, 3, 1)
	box1 = box1.view(box1.shape[0], -1, box1.shape[3])
	box2 = box2.view(box2.shape[0], -1, box2.shape[3]) 
	boxes = torch.cat([box1, box2], 1).numpy()
	obj_mask = boxes[:, :, 4] >=obj_threshold
	selected_boxes = []
	
	for box, mask in zip(boxes, obj_mask):
		mask = mask.reshape(-1)
		this_boxes = []
		temp_boxes = []
		for i in range(mask.shape[0]):	
			if mask[i]:
				temp_boxes.append(box[i])
		
		box = sorted(temp_boxes, key=lambda a:a[-2])
		box_list = [[] for _ in range(20)]
		for i in range(len(box)):
			box_list[int(box[i][-1])].append(box[i]) 
			
		for sorted_ in box_list:
			while len(sorted_) > 0:
				this_box = sorted_[0]
				del sorted_[0]
				sorted_new = []
				for i in range(len(sorted_)):
					if not IoU(this_box, sorted_[i]) > iou_threshold:
						sorted_new.append(sorted_[i])

				sorted_ = sorted_new
				this_box[4] = 1
				this_boxes.append(this_box)
		selected_boxes.append(torch.tensor(this_boxes))
	return selected_boxes
