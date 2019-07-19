import torch 
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
from PIL import Image
import numpy as np
from absl import flags, app

from class_names import class_names

FLAGS = flags.FLAGS

flags.DEFINE_float('iou_threshold', 0.6, 
									'--minimum iou for box rejection')
flags.DEFINE_float('obj_threshold', 0.25, 
									'--minimum score for box consideration')

np.random.seed(100)
color_dict = np.random.rand(20,3)

def visualize_bbox(image, label_tensor, save_path=None):
	object_mask = label_tensor[:,:,4] == 1
	object_mask = object_mask.unsqueeze(dim=-1).expand(label_tensor.shape)
	#arrange the values into a tensor of box coordinates 
	label_tensor = label_tensor[object_mask].view((-1, 6))
	scale_x, scale_y = image.size(1), image.size(2)
	centers = label_tensor[:,0:2]
	sizes = label_tensor[:,2:4]
	label_tensor[:,0:2] = centers - sizes / 2
	label_tensor[:,0:2] = label_tensor[:,0:2] * torch.tensor([scale_x, scale_y], dtype=torch.float32)
	label_tensor[:,2:4] = label_tensor[:,2:4] * torch.tensor([scale_x, scale_y], dtype=torch.float32)
	label_tensor = label_tensor.to(torch.int16)
	fig, ax = plt.subplots(1)
	ax.imshow(image.permute(1,2,0))

	for i in range(label_tensor.size(0)):
		ax.add_patch(patches.Rectangle((label_tensor[i, 0], label_tensor[i, 1]),
														label_tensor[i, 2], label_tensor[i, 3], 
														color=color_dict[label_tensor[i, 5]], fill=False))
		ax.text(label_tensor[i, 0], label_tensor[i, 1], class_names[label_tensor[i, 5]] ,fontsize=20, color=color_dict[label_tensor[i, 5]])
	if save_path is not None:
		plt.savefig(save_path)
	
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

def non_max_suppression( predicted_tensor):
	#grid coordinates to add to predicted offset
	predicted_tensor = predicted_tensor.cpu()
	x = torch.tensor(torch.arange(7))
	y = torch.tensor(torch.arange(7))
	grid_x, grid_y = torch.meshgrid(x, y)
	box1, box2, classes = predicted_tensor.split([5, 5, 20], 1)
	classes = torch.argmax(classes, 1, keepdim=True).type(torch.float)
	grid = torch.cat([grid_x.unsqueeze(0), grid_y.unsqueeze(0)], 0).type(torch.float)
	box1_coords, box1_dims, box1_confs = box1.split([2,2,1], 1)
	box2_coords, box2_dims, box2_confs = box2.split([2,2,1], 1)
	box1_coords = (box1_coords + grid)/7
	box2_coords = (box2_coords + grid)/7
	box1 = torch.cat([box1_coords, box1_dims, classes, box1_confs], dim=1).permute(0, 2, 3, 1)
	box2 = torch.cat([box2_coords, box2_dims, classes, box2_confs], dim=1).permute(0, 2, 3, 1)
	box1 = box1.view(box1.shape[0], -1, box1.shape[3])
	box2 = box2.view(box2.shape[0], -1, box2.shape[3]) 
	boxes = torch.cat([box1, box2], 1).numpy()
	obj_mask = boxes[:, :, 5] >= FLAGS.obj_threshold
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
			box_list[int(box[i][-2])].append(box[i]) 
		for sorted_ in box_list:
			while len(sorted_) > 0:
				this_box = sorted_[0]
				del sorted_[0]
				sorted_new = []
				for i in range(len(sorted_)):
					if not IoU(this_box, sorted_[i]) > FLAGS.iou_threshold:
						sorted_new.append(sorted_[i])

				sorted_ = sorted_new
				this_boxes.append(this_box)
		selected_boxes.append(torch.tensor(this_boxes))
	return selected_boxes
	
if __name__ == '__main__':
	random_prediction = torch.rand([1, 30, 7, 7], device='cuda')	
	def run_(argv):
		visualize_bbox('test.png', non_max_suppression(random_prediction)[0])
	app.run(run_)
