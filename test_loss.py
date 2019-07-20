import torch
import sys
from absl import flags, app

from yolov1loss import yolov1loss
from yolov1 import yolonet
from utils import *
from load_labels import *

def test(argv):
	train, test = VOCDataset(16, 16)
	trainiter = iter(test)
	x, y = next(trainiter)
	#visualize_bbox(x[0], y[0], save_path='test.png')
	device = 'cuda'
	net = yolonet().to(device)
	
	loss = yolov1loss(lambda_coord=0.5, lambda_noobj=0.5, device=device)
	#labels = torch.ones([int(sys.argv[1]), 6, 7, 7] , device=device)
	labels = y[0].unsqueeze(0).to(device)
	#random_image = torch.rand([1, 3, 448, 448], device=device)
	random_image = x[0].unsqueeze(0).to(device)
	with torch.no_grad():
		preds = net(random_image)
	#	suppressed_preds = non_max_suppression(preds)
	#	visualize_bbox( random_image[0].to('cpu'), suppressed_preds[0], save_path='random_preds.png')	
	#print (preds)
	print(loss(preds, labels))

if __name__ == '__main__':
	app.run(test)
