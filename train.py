import torch
from torchsummary import summary
from yolov1 import yolonet
from load_labels import VOCDataset
from yolov1loss import yolov1loss

def train(**kwargs):
	#make a network object
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	net = yolonet()
	net = net.to(device)
	print(summary(net, (3,448,448))
	
	traindata, valdata = VOCDataset()
	train_loss, train_acc, val_loss, val_acc = [],[],[],[]
	for _ in range(kwargs[n_epochs]):
		trianiter = iter(traindata)
		try:
			x, y = trainiter.next()
			x = x.to(device)
			y = y.to(device)
			y_hat = net(x)
			box1, box2, logits = torch.split(y_hat, [5, 5, 20], axis=1)
			
		
	
		

