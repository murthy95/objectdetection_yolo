import torch 
import torch.nn as nn
from utils import *

class yolov1loss(nn.Module):
	def __init__(self):
		super(yolov1loss, self).__init__()
		pass
	
	def forward(self, x, labels):
		'''
	  args x output from the network tensor of shape 7, 7, 20 
				labels labels tensor shape 7, 7, 6

		'''
		#evaluating coordinate loss
		box1, box2, class_predictions = x.split([5, 5, 20], dim=1)
		#only one box is selected for evalutaing loss box with maximum iou score with ground truth labels is selected 
		
		
		
