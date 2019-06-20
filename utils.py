import torch 
import matplotlib.pyplot as plt 
from PIL import Image
import numpy as np

np.random.seed(100)
color_dict = np.random.randint(0, 255, 20*3).reshape(-1, 3)

def visualize_bbox(image, label_tensor, save_path=None):
	object_mask = label_tensor[:,:,4] == 1
	object_mask = object_mask.unsqueeze(dim=-1).expand(label_tensor.shape)
	#arrange the values into a tensor of box coordinates 
	objects = label_tensor[object_mask].view((6,-1)).transpose(0,1) 
	scale_x, scale_y = image.size(0), image.size(1)
	centers = label_tensor[:,0:2]
	sizes = label_tensor[:,2:4]
	class_labels = label_tensor[:,5]
	label_tensor[:,0:2] = centers - sizes / 2
	label_tensor[:,0:2] = label_tensor[:,0:2] * torch.tensor([scale_x, scale_y])
	label_tensor[:,2:4] = label_tensor[:,0:2] * torch.tensor([scale_x, scale_y])
	label_tensor.to(torch.int16)

	plot = plt.plot(image.data)
	for i in range(label_tensor.size(0)):
		plt.patches.Rectangle((label_tensor[i, 0], label_tensor[i, 1]),
														label_tensor[i, 2], label_tensor[i, 3], 
														color=color_dict[label_tensor[i, 4]])
	if save_path is not None:
		plt.savefig(save_path)
	


		
	
	
	
	
