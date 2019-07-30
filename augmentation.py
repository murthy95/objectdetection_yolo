'''
All the augmentations assume the image as numpy array 
and bbox coordinates as numpy array 
image is numpy array with shape [height, width, channels]
bbox is array with shape [nboxes, 5]
the 5 channels represent [xmin, ymin, xmax, ymax, class]
output of the transformation call is same shape as the inputs

'''

import numpy as np
import torch
from utils import to_center_dims_format, to_min_max_format
import cv2

def translation(image, bboxes, max_trans=0.2):
	shape = image.shape
	indexy, indexx = np.random.randint(-max_trans * shape[0]/2, 
				shape[0]/2 *max_trans), np.random.randint(-max_trans * shape[1]/2, max_trans *shape[1]/2)
	miny = max(indexy, 0) 
	maxy = min(shape[0], shape[0] + indexy)
	minx = max(0, indexx) 
	maxx = min(shape[1], shape[1] + indexx)

	return random_cropping(image, bboxes, np.array([minx, miny, maxx, maxy]))

def flip(image, bboxes):
	'''
		flips the image and bounding boxes in the direction specified 
	'''
	if np.random.rand() < 0.5 :
		#horizontal flip
		image = np.flip(image, 1)
		bboxes[:,0] = image.shape[1] - bboxes[:, 0]
		bboxes[:,2] = image.shape[1] - bboxes[:, 2]
		temp = bboxes[:,0]
		bboxes[:,0] = bboxes[:,2]
		bboxes[:,2] = temp
	else:
		#vertical flip
		image = np.flip(image, 0)
		bboxes[:,1] = image.shape[0] - bboxes[:, 1]
		bboxes[:,3] = image.shape[0] - bboxes[:, 3]
		temp = bboxes[:,1]
		bboxes[:,1] = bboxes[:,3]
		bboxes[:,3] = temp
	return image, bboxes

def random_cropping(image, bboxes, dims=None, max_trans=0.2):
	'''
		crop image boundaries specified by dims and resize to orginal shape
	'''
	shape = image.shape[:2]
	if dims is None:
		xmin = np.random.randint(0, max_trans * shape[1])
		xmax = np.random.randint(shape[1] - max_trans*shape[1], shape[1])
		ymin = np.random.randint(0, max_trans * shape[0])
		ymax = np.random.randint(shape[0] - max_trans*shape[0], shape[0])
		dims = np.array([xmin, ymin, xmax, ymax])
	
	maxs = np.tile(dims[2:], 2)
	mins = np.tile(dims[:2], 2)
	scale = np.tile([shape[1], shape[0]], 2)
	
	image = image[dims[1]:dims[3], dims[0]:dims[2], :]
	bboxes[:,:4] = bboxes[:,:4].clip(mins, maxs)
	bboxes = bboxes[np.where(bbox_area(bboxes) > 0)[0]]
	
	image = cv2.resize(image, (shape[1], shape[0]))
	bboxes[:, :4] = ((bboxes[:, :4] - mins)	/ (maxs - mins)) * scale
	return image, bboxes

def hsv_augmentation(image, bboxes):
	exposure = np.random.rand()*1.5
	saturation = np.random.rand()*1.5
	hsvimage = cv2.cvtColor(np.flip(image, 2), cv2.COLOR_BGR2HSV).astype(float)
	hsvimage *= np.array([1.0, saturation, exposure])
	hsvimage = hsvimage.clip( [0,0,0], [179, 255, 255])
	return np.flip(cv2.cvtColor(hsvimage.astype(np.uint8), cv2.COLOR_HSV2BGR), 2), bboxes

def bbox_area(bboxes):
	cboxes = to_center_dims_format(bboxes)
	return cboxes[:,2]*cboxes[:,3]

def convert_to_tensor(bboxes):
	return [torch.tensor(bbox, dtype=torch.float) for bbox in bboxes]

def main():
	image = np.asarray(Image.open(sys.argv[1]).convert('RGB'))
	print ('Image Shape : {}'.format(image.shape))
	bboxes = np.array([[5., 138., 246., 382., 1., 7.],
										[196., 1., 558., 378., 1., 11.]])
			
	image, bboxes = hsv_augmentation(image, bboxes)
	bboxes = to_center_dims_format(bboxes)
	visualize_bbox(image, torch.tensor(bboxes, dtype=torch.float32), 'augmentation_check.png')

if __name__ == '__main__':
	from PIL import Image
	import sys
	from utils import *
	main()
