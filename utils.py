import torch 
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
from PIL import Image
import numpy as np
from class_names import class_names

np.random.seed(100)
color_dict = np.random.rand(20,3)

def visualize_bbox(image, label_tensor, save_path=None):
	object_mask = label_tensor[:,:,4] == 1
	object_mask = object_mask.unsqueeze(dim=-1).expand(label_tensor.shape)
	#arrange the values into a tensor of box coordinates 
	label_tensor = label_tensor[object_mask].view((-1, 6))
	print(label_tensor) 
	scale_x, scale_y = image.size(1), image.size(2)
	centers = label_tensor[:,0:2]
	sizes = label_tensor[:,2:4]
	label_tensor[:,0:2] = centers - sizes / 2
	label_tensor[:,0:2] = label_tensor[:,0:2] * torch.tensor([scale_x, scale_y], dtype=torch.float32)
	label_tensor[:,2:4] = label_tensor[:,2:4] * torch.tensor([scale_x, scale_y], dtype=torch.float32)
	label_tensor = label_tensor.to(torch.int16)
	print (label_tensor)
	fig, ax = plt.subplots(1)
	ax.imshow(image.permute(1,2,0))

	for i in range(label_tensor.size(0)):
		ax.add_patch(patches.Rectangle((label_tensor[i, 0], label_tensor[i, 1]),
														label_tensor[i, 2], label_tensor[i, 3], 
														color=color_dict[label_tensor[i, 5]], fill=False))
		ax.text(label_tensor[i, 0], label_tensor[i, 1], class_names[label_tensor[i, 5]] ,fontsize=20, color=color_dict[label_tensor[i, 5]])
	if save_path is not None:
		plt.savefig(save_path)
	


		
	
	
	
	
