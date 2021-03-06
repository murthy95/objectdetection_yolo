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

def to_min_max_format(bboxes):
	minmax = np.zeros(bboxes.shape)
	minmax[:, :] = bboxes[:, :]
	minmax[:, :2] -= minmax[:, 2:4]/2
	minmax[:, 2:4] += minmax[:, :2] 
	return minmax	
	
def to_center_dims_format(bboxes):
	centerdims = np.zeros(bboxes.shape)
	centerdims[:, :] = bboxes[:, :]
	centerdims[:, 2:4] -= centerdims[:, :2] 
	centerdims[:, :2] += centerdims[:, 2:4]/2
	return centerdims	

def visualize_bbox(image, label_tensor, save_path=None):
	fig, ax = plt.subplots(1)
	if isinstance(image, torch.Tensor):
		ax.imshow(image.permute(1,2,0))
		scale_x, scale_y = image.size(1), image.size(2)
	else:
		ax.imshow(image)
		scale_x, scale_y = image.shape[0], image.shape[1]

	if len(list(label_tensor.shape)) <  2:
		if save_path is not None :
			plt.savefig(save_path) 
			return 
		if torch.max(label_tensor) > 1:
			label_tensor[:, :2] /= torch.tensor([scale_x, scale_y], dtype=torch.float32)
			label_tensor[:, 2:4] /= torch.tensor([scale_x, scale_y], dtype=torch.float32)

	if len(list(label_tensor.shape)) == 3:
		print ('recieved true labels')
		x = torch.tensor(torch.arange(7))
		y = torch.tensor(torch.arange(7))
		grid_x, grid_y = torch.meshgrid(x, y)
		grid = torch.cat([grid_y.unsqueeze(0), grid_x.unsqueeze(0)], 0).type(torch.float)
		#print (grid)
		label_tensor[:2,:,:] += grid 
		label_tensor[:2,:,:] /= 7
		label_tensor = label_tensor.permute(1, 2, 0)
		object_mask = label_tensor[:, :, 4] == 1
		object_mask = object_mask.unsqueeze(dim=-1).expand(label_tensor.shape)
		#arrange the values into a tensor of box coordinates 
		label_tensor = label_tensor[object_mask].view((-1, 6))
	
	centers = label_tensor[:,:2]
	sizes = label_tensor[:,2:4]
	label_tensor[:,:2] = centers - sizes / 2
	label_tensor[:,:2] = label_tensor[:,:2] * torch.tensor([scale_x, scale_y], dtype=torch.float32)
	label_tensor[:,2:4] = label_tensor[:,2:4] * torch.tensor([scale_x, scale_y], dtype=torch.float32)
	label_tensor = label_tensor.to(torch.int16)
	for i in range(label_tensor.size(0)):
		ax.add_patch(patches.Rectangle((label_tensor[i, 0], label_tensor[i, 1]),
														label_tensor[i, 2], label_tensor[i, 3], 
														color=color_dict[label_tensor[i, 5]], fill=False))
		ax.text(label_tensor[i, 0], label_tensor[i, 1], class_names[label_tensor[i, 5]] ,fontsize=20, color=color_dict[label_tensor[i, 5]])
	if save_path is not None:
		plt.savefig(save_path)
	else :
		fig.canvas.draw()
		return np.array(fig.canvas.renderer.buffer_rgba()) 

def visualize_bbox_labels(image, label_tensor, bboxes, save_path):
	fig, (ax1, ax2) = plt.subplots(1, 2)
	if isinstance(image, torch.Tensor):
		ax1.imshow(image.permute(1,2,0))
		ax2.imshow(image.permute(1,2,0))
		scale_x, scale_y = image.size(1), image.size(2)
	else:
		ax1.imshow(image)
		ax2.imshow(image)
		scale_x, scale_y = image.shape[0], image.shape[1]

	if len(list(bboxes.shape)) <  2:
		if save_path is not None :
			plt.savefig(save_path) 
			return 
		if torch.max(label_tensor) > 1:
			label_tensor[:, :2] /= torch.tensor([scale_x, scale_y], dtype=torch.float32)
			label_tensor[:, 2:4] /= torch.tensor([scale_x, scale_y], dtype=torch.float32)

	if len(list(label_tensor.shape)) == 3:
		print ('recieved true labels')
		x = torch.tensor(torch.arange(7))
		y = torch.tensor(torch.arange(7))
		grid_x, grid_y = torch.meshgrid(x, y)
		grid = torch.cat([grid_y.unsqueeze(0), grid_x.unsqueeze(0)], 0).type(torch.float)
		label_tensor[:2,:,:] += grid 
		label_tensor[:2,:,:] /= 7
		label_tensor = label_tensor.permute(1, 2, 0)
		object_mask = label_tensor[:, :, 4] == 1
		object_mask = object_mask.unsqueeze(dim=-1).expand(label_tensor.shape)
		#arrange the values into a tensor of box coordinates 
		label_tensor = label_tensor[object_mask].view((-1, 6))
	
	centers = label_tensor[:,:2]
	sizes = label_tensor[:,2:4]
	label_tensor[:,:2] = centers - sizes / 2
	label_tensor[:,:2] = label_tensor[:,:2] * torch.tensor([scale_x, scale_y], dtype=torch.float32)
	label_tensor[:,2:4] = label_tensor[:,2:4] * torch.tensor([scale_x, scale_y], dtype=torch.float32)
	label_tensor = label_tensor.to(torch.int16)
	
	for i in range(label_tensor.size(0)):
		ax1.add_patch(patches.Rectangle((label_tensor[i, 0], label_tensor[i, 1]),
														label_tensor[i, 2], label_tensor[i, 3], 
														color=color_dict[label_tensor[i, 5]], fill=False))
		ax1.text(label_tensor[i, 0], label_tensor[i, 1], class_names[label_tensor[i, 5]] ,fontsize=20, color=color_dict[label_tensor[i, 5]])
	
	centers = bboxes[:,:2]
	sizes = bboxes[:,2:4]
	bboxes[:,:2] = centers - sizes / 2
	bboxes[:,:2] = bboxes[:,:2] * torch.tensor([scale_x, scale_y], dtype=torch.float32)
	bboxes[:,2:4] = bboxes[:,2:4] * torch.tensor([scale_x, scale_y], dtype=torch.float32)
	bboxes = bboxes.to(torch.int16)
	for i in range(bboxes.size(0)):
		ax2.add_patch(patches.Rectangle((bboxes[i, 0],bboxes[i, 1]),
														bboxes[i, 2], bboxes[i, 3], 
														color=color_dict[bboxes[i, 5]], fill=False))
		ax2.text(bboxes[i, 0], bboxes[i, 1], class_names[bboxes[i, 5]] ,fontsize=20, color=color_dict[bboxes[i, 5]])

	if save_path is not None:
		plt.savefig(save_path)
	else :
		fig.canvas.draw()
		return np.array(fig.canvas.renderer.buffer_rgba()) 


def IoU(bb1, bb2):
	'''
	bb1 : input bounding box 1 the coordinates corresponds to 
	      center_x center_y width height
	bb2 : bounding box 2 coordinates in the same order as bb1
	'''
	minx = max(bb1[0] - bb1[2] / 2, bb2[0] - bb2[2] / 2)
	maxx = min(bb1[0] + bb1[2] / 2, bb2[0] + bb2[2] / 2)
	miny = max(bb1[1] - bb1[3] / 2, bb2[1] - bb2[3] / 2)
	maxy = min(bb1[1] + bb1[3] / 2, bb2[1] + bb2[3] / 2)
	diffx = (maxx - minx) #- (bb1[2] + bb2[2]) 
	diffy = (maxy - miny) #- (bb1[3] + bb2[3])
	if diffx > 0  and diffy >  0 :
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
	box1 = torch.sigmoid(box1)
	box2 = torch.sigmoid(box2)
	classes = torch.argmax(classes, 1, keepdim=True).type(torch.float)
	grid = torch.cat([grid_y.unsqueeze(0), grid_x.unsqueeze(0)], 0).type(torch.float)
	box1_coords, box1_dims, box1_confs = box1.split([2,2,1], 1)
	box2_coords, box2_dims, box2_confs = box2.split([2,2,1], 1)
	box1_coords = (box1_coords + grid)/7
	box2_coords = (box2_coords + grid)/7
	box1 = torch.cat([box1_coords, box1_dims, box1_confs, classes], dim=1).permute(0, 2, 3, 1)
	box2 = torch.cat([box2_coords, box2_dims, box2_confs, classes], dim=1).permute(0, 2, 3, 1)
	box1 = box1.view(box1.shape[0], -1, box1.shape[3])
	box2 = box2.view(box2.shape[0], -1, box2.shape[3]) 
	boxes = torch.cat([box1, box2], 1).numpy()
	obj_mask = boxes[:, :, 4] >= FLAGS.obj_threshold
	selected_boxes = []
	
	for box, mask in zip(boxes, obj_mask):
		mask = mask.reshape(-1)
		this_boxes = []
		temp_boxes = []
		for i in range(mask.shape[0]):	
			if mask[i]:
				temp_boxes.append(box[i])
		
		box = sorted(temp_boxes, key=lambda a:a[-2])
		box = box[::-1]
		box_list = [[] for _ in range(20)]
		for i in range(len(box)):
			box_list[int(box[i][-1])].append(box[i]) 
			
		for sorted_ in box_list:
			while len(sorted_) > 0:
				this_box = sorted_[0]
				del sorted_[0]
				sorted_new = []
				for i in range(len(sorted_)):
					if IoU(this_box, sorted_[i]) < FLAGS.iou_threshold:
						sorted_new.append(sorted_[i])

				sorted_ = sorted_new
				this_box[4] = 1
				this_boxes.append(this_box)
		selected_boxes.append(torch.tensor(this_boxes))
	return selected_boxes
	
if __name__ == '__main__':
	random_prediction = torch.rand([1, 30, 7, 7], device='cuda')	
	def run_(argv):
		visualize_bbox('test.png', non_max_suppression(random_prediction)[0])
	app.run(run_)
