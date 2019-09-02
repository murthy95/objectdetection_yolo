import torch 
import torch.nn as nn
from utils import *


class yolov1loss(nn.Module):

	def __init__(self, lambda_coord, lambda_noobj, device ):
		super(yolov1loss, self).__init__()
		self.lambda_coord = lambda_coord
		self.lambda_noobj = lambda_noobj
		self.device = device

	def forward(self, x, labels):
		'''
	  args x 		:output from the network tensor of shape 7, 7, 20 
				labels :labels tensor shape 7, 7, 6

		'''
		#evaluating coordinate loss
		box1, box2, class_predictions = x.split([5, 5, 20], dim=1)
		box1 = torch.sigmoid(box1)
		box2 = torch.sigmoid(box2)
		label_coords, label_dims, object_mask, class_labels = labels.split([2, 2, 1, 1], dim=1)
		
#		class_predictions = class_predictions.permute(0,2,3,1)
#		indices = torch.nonzero(object_mask[:,0,:,:])
#		class_predictions = torch.take(class_predictions, indices)
		
#		class_labels = torch.take(class_labels.type(torch.int64)[:,0,:,:], indices)
		
		class_predictions = torch.softmax(class_predictions, dim=1)
		class_labels = class_labels.type(torch.int64)
#		loss = torch.mean((class_predictions - class_labels)**2)
#		loss = torch.mean(temp)
#		print (class_predictions.shape, class_labels.shape)
#		loss = self.ce_loss(class_predictions, class_labels)		
		onehotlabels = torch.zeros([labels.shape[0], 20, labels.shape[2],
														labels.shape[3]], device=self.device).scatter(1, class_labels, object_mask)
		loss = torch.mean(torch.sum((object_mask*class_predictions - onehotlabels)**2, dim=[1,2,3]))	

			
		#1ij obj is 1 if object is present in the cell and confidence of jth box is highest of all boxes 	
		box1_coords, box1_dims, box1_confs = box1.split([2,2,1], dim=1)
		box2_coords, box2_dims, box2_confs = box2.split([2,2,1], dim=1)
		box1_dims = torch.sqrt(box1_dims + 1e-6)
		box2_dims = torch.sqrt(box2_dims + 1e-6)
		label_dims = torch.sqrt(label_dims + 1e-6)
		box_confs = torch.cat([box1_confs, box2_confs], 1)
		argmax_conf = torch.argmax(box_confs, 1, keepdim=True).detach()
		argmax_conf = argmax_conf.type(torch.float)
		
		
		final_box_coords = (1-argmax_conf)*box1_coords + argmax_conf*box2_coords
		loss += self.lambda_coord * (torch.mean(torch.sum((
														object_mask*final_box_coords - label_coords)**2, dim=[1,2,3])))
		
		final_box_dims = (1-argmax_conf)*box1_dims + argmax_conf*box2_dims
		loss += self.lambda_coord * (torch.mean(torch.sum((        		
														object_mask*final_box_dims - label_dims)**2, dim=[1,2,3])))
		
		final_box_obj = (1-argmax_conf)*box1_confs + argmax_conf*box2_confs
	#	selected_box_obj = torch.take(final_box_obj, object_mask.nonzero()) 
	#	loss += torch.mean((selected_box_obj - 1) **2)
		object_mask = object_mask.type(torch.float)
		loss += torch.mean(torch.sum((object_mask*final_box_obj - object_mask)**2, dim=[1,2,3]))
		
		#no obj loss
		rejected_boxes =(1-argmax_conf)*box2_confs + argmax_conf*box1_confs
#		loss += self.lambda_noobj * torch.mean(torch.sum(rejected_boxes)**2) 
#		loss += self.lambda_noobj * torch.mean((torch.take(final_box_obj, (1-object_mask).nonzero()))**2)
# 
		loss += self.lambda_noobj*torch.mean(torch.sum(
												((1-object_mask)*final_box_obj)**2 
												+ (rejected_boxes)**2, dim=[1,2,3]))
		return loss
