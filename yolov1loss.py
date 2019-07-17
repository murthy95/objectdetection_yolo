import torch 
import torch.nn as nn
from utils import *

class yolov1loss(nn.Module):
	def __init__(self, lambda_coord, lambda_noobj, threshold=0.25):
		super(yolov1loss, self).__init__()
		self.lamda_coord = lambda_coord
		self.lamda_noobj = lambda_noobj
		self.objectivity_threshold = threshold
	
	def forward(self, x, labels):
		'''
	  args x 		:output from the network tensor of shape 7, 7, 20 
				labels :labels tensor shape 7, 7, 6

		'''
		#evaluating coordinate loss
		box1, box2, class_predictions = x.split([5, 5, 20], dim=1)
		label_coords, label_dims, obj_mask, class_labels = labels.split([2,2, 1, 1], dim=1)
		
		onehotlabels = torch.zeros(labels.shape[0], 20, labels.shape[2],
														labels.shape[2]).scatter(1, class_labels, object_mask)
		loss = torch.mean(class_predictions - onehotlabels)
		
		#1ij obj is 1 if object is present in the cell and confidence of jth box is highest of all boxes 	
		box1_coords, box1_dims, box1_confs = box1.split([2,2,1], dim=1)
		box2_coords, box2_dims, box2_confs = box2.split([2,2,1], dim=1)
		box1_dims = torch.sqrt(box1_dims)
		box2_dims = torch.sqrt(box2_dims)
		label_dims = torch.sqrt(label_dims)
	
		box_confs = torch.cat([box1[:, -1, :, :], box2[:, -1, :, :]], 1)
		argmax_conf = torch.argmax(box_conf, 1)
		box1_mask = object_mask * ( argmax_conf == 0 )
		box2_mask = object_mask * ( argmax_conf == 1 )
		
		final_box_coords = (1-argmax_conf)*box1_coords + argmax_conf*box2_coords
		loss += self.lambda_coord * (torch.mean(torch.square(
														obj_mask*final_box_coords - label_coords)))
		final_box_dims = (1-argmax_conf)*box1_dims + argmax_conf*box2_dims
		loss += self.lambda_coord * (torch.mean(torch.square(        		
														obj_mask*final_box_dims - label_dims)))

		box1_confs = box1_confs.clamp(min=self.objectivity_threshold, max=1)
		box2_confs = box2_confs.clamp(min=self.objectivity_threshold, max=1)	 
		final_box_obj = (1-argmax_conf)*box1_confs+ argmax_conf*box2_confs 
		loss += torch.mean(torch.square(object_mask*final_box_obj - object_mask))

		#no obj loss
		loss += torch.mean(torch.square((1-object_mask)*final_box_obj - object_mask))
		
		return loss
